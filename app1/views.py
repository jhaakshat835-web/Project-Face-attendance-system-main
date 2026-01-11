import os
import base64
import threading
import time
from datetime import datetime, timedelta

from django.conf import settings
from django.shortcuts import render, redirect
from django.core.files.base import ContentFile
from django.utils import timezone
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError

from .models import Student, Attendance, CameraConfiguration

# ================= LAZY LOADING GLOBALS =================
# These variables start as None and are only loaded when needed.
device = None
mtcnn = None
resnet = None

def load_models():
    """
    Loads heavy AI libraries and models only when called.
    This prevents the server from crashing during startup.
    """
    global device, mtcnn, resnet
    
    # Only import heavy libraries here
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if mtcnn is None:
        mtcnn = MTCNN(keep_all=True, device=device)
        
    if resnet is None:
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        
    return device, mtcnn, resnet

# ================= FACE ENCODING =================
def detect_and_encode(image):
    # Import standard libraries locally to save startup memory
    import cv2
    import numpy as np
    import torch

    # Ensure models are loaded
    local_device, local_mtcnn, local_resnet = load_models()

    with torch.no_grad():
        boxes, _ = local_mtcnn.detect(image)
        encodings = []

        if boxes is None:
            return encodings

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            h, w, _ = image.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = image[y1:y2, x1:x2]
            if face.size == 0: continue

            face = cv2.resize(face, (160, 160))
            face = np.transpose(face, (2, 0, 1))
            face_tensor = torch.tensor(face).float().to(local_device) / 255.0
            face_tensor = face_tensor.unsqueeze(0)

            encoding = local_resnet(face_tensor).detach().cpu().numpy().flatten()
            encodings.append(encoding)

        return encodings

# ================= LOAD KNOWN FACES =================
def encode_uploaded_images():
    import cv2
    import numpy as np
    
    known_face_encodings = []
    known_face_names = []

    students = Student.objects.filter(authorized=True)
    
    print(f"Loading {students.count()} authorized students...")

    for student in students:
        if not student.image: continue
        
        image_path = os.path.join(settings.MEDIA_ROOT, str(student.image))
        try:
            image = cv2.imread(image_path)
            if image is None: continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = detect_and_encode(image_rgb)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(student.name)
        except Exception as e:
            print(f"Error loading {student.name}: {e}")

    return np.array(known_face_encodings), known_face_names

# ================= FACE RECOGNITION =================
def recognize_faces(known_encodings, known_names, test_encodings, threshold=0.6):
    import numpy as np
    
    names = []
    if len(known_encodings) == 0:
        return ["Unknown"] * len(test_encodings)

    for test_encoding in test_encodings:
        distances = np.linalg.norm(known_encodings - test_encoding, axis=1)
        min_index = np.argmin(distances)

        if distances[min_index] < threshold:
            names.append(known_names[min_index])
        else:
            names.append("Not Recognized")

    return names

# ================= CAPTURE & RECOGNIZE =================
def capture_and_recognize(request):
    import cv2
    
    # 1. Load models first (This is where the heavy lifting happens)
    try:
        load_models()
    except Exception as e:
        messages.error(request, f"Error loading AI models: {e}")
        return redirect("home")

    # 2. Load student data
    known_encodings, known_names = encode_uploaded_images()
    
    if len(known_encodings) == 0:
        messages.error(request, "No authorized students found with photos.")
        return redirect("home")

    attendance_event = threading.Event()
    threads = []
    stop_events = []
    found_name = [None] 

    def camera_thread(cam_config, stop_event):
        import cv2 # Ensure cv2 is available in thread
        
        src = cam_config.camera_source
        if str(src).isdigit(): src = int(src)
        
        cap = cv2.VideoCapture(src)
        start_time = time.time()

        while not stop_event.is_set():
            if time.time() - start_time > 30:
                break

            ret, frame = cap.read()
            if not ret: break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            test_encodings = detect_and_encode(rgb)

            if test_encodings:
                names = recognize_faces(known_encodings, known_names, test_encodings, cam_config.threshold)

                for name in names:
                    if name != "Not Recognized":
                        student = Student.objects.filter(name=name).first()
                        if student:
                            # Mark Attendance
                            att, created = Attendance.objects.get_or_create(
                                student=student,
                                date=datetime.now().date()
                            )
                            if created:
                                att.mark_checked_in()
                            elif att.check_in_time and not att.check_out_time:
                                if timezone.now() >= att.check_in_time + timedelta(minutes=1):
                                    att.mark_checked_out()
                            
                            found_name[0] = name
                            attendance_event.set()
                            stop_event.set()
                            break
            
            # Note: cv2.imshow lines removed for server stability

        cap.release()
        cv2.destroyAllWindows()

    cams = CameraConfiguration.objects.all()
    if not cams:
        messages.error(request, "No cameras configured.")
        return redirect("home")

    for cam in cams:
        stop = threading.Event()
        stop_events.append(stop)
        t = threading.Thread(target=camera_thread, args=(cam, stop))
        threads.append(t)
        t.start()

    attendance_event.wait(timeout=35)

    for e in stop_events: e.set()
    for t in threads: t.join()

    if found_name[0]:
        messages.success(request, f"Attendance marked for: {found_name[0]}")
    else:
        messages.warning(request, "Time out: No face recognized.")

    return redirect("student_attendance_list") 

# ================= VIEWS FOR LIST AND ATTENDANCE =================

def student_list(request):
    students = Student.objects.all().order_by('name')
    return render(request, "student_list.html", {"students": students})


def student_attendance_list(request):
    records = Attendance.objects.all().order_by("-date", "-check_in_time")
    return render(request, "student_attendance_list.html", {"student_attendance_data": records})


# ================= STUDENT REGISTRATION =================
def capture_student(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        reg_no = request.POST.get("registration_number")
        branch = request.POST.get("branch")
        year = request.POST.get("year")
        image_data = request.POST.get("image_data")

        if image_data:
            try:
                format, imgstr = image_data.split(';base64,') 
                ext = format.split('/')[-1] 
                image_file = ContentFile(base64.b64decode(imgstr), name=f"{name}_{timezone.now().timestamp()}.{ext}")

                Student.objects.create(
                    name=name, email=email, registration_number=reg_no,
                    branch=branch, year=year, image=image_file, authorized=True
                )
                return redirect("selfie_success")
            except Exception as e:
                messages.error(request, f"Error: {e}")

    return render(request, "capture_student.html")

def selfie_success(request):
    return render(request, "selfie_success.html")

# ================= HOME & AUTH =================
def home(request):
    return render(request, "home.html", {
        "total_students": Student.objects.count(),
        "total_attendance": Attendance.objects.count(),
        "total_check_ins": Attendance.objects.filter(check_in_time__isnull=False).count(),
        "total_check_outs": Attendance.objects.filter(check_out_time__isnull=False).count(),
        "total_cameras": CameraConfiguration.objects.count()
    })

def user_login(request):
    if request.method == "POST":
        user = authenticate(request, username=request.POST.get("username"), password=request.POST.get("password"))
        if user:
            login(request, user)
            return redirect("home")
        messages.error(request, "Invalid credentials")
    return render(request, "login.html")

def user_logout(request):
    logout(request)
    return redirect("home")

# ================= STUDENT DETAIL VIEW =================
def student_detail(request, pk):
    student = Student.objects.filter(pk=pk).first()
    
    if not student:
        messages.error(request, "Student not found")
        return redirect("student-list")

    records = Attendance.objects.filter(student=student).order_by("-date")
    
    return render(request, "student_detail.html", {
        "student": student, 
        "attendance_records": records
    })

# ================= AUTHORIZE & DELETE =================
def student_authorize(request, pk):
    student = Student.objects.filter(pk=pk).first()
    if student:
        student.authorized = True
        student.save()
        messages.success(request, "Student authorized successfully!")
    return redirect("student-list") 

def student_delete(request, pk):
    student = Student.objects.filter(pk=pk).first()
    if student:
        student.delete()
        messages.success(request, "Student deleted successfully!")
    return redirect("student-list") 

# ================= CAMERA SETTINGS =================
def camera_config_list(request):
    cameras = CameraConfiguration.objects.all()
    return render(request, "camera_config_list.html", {"cameras": cameras})

def camera_config_create(request):
    if request.method == "POST":
        name = request.POST.get("name")
        source = request.POST.get("camera_source")
        try:
            threshold = float(request.POST.get("threshold", 0.6))
        except ValueError:
            threshold = 0.6
        
        CameraConfiguration.objects.create(
            name=name,
            camera_source=source,
            threshold=threshold
        )
        messages.success(request, "Camera added successfully!")
        return redirect("camera_config_list")

    return render(request, "camera_config_create.html")

def camera_config_update(request, pk):
    camera = CameraConfiguration.objects.filter(pk=pk).first()
    if not camera:
        return redirect("camera_config_list")

    if request.method == "POST":
        camera.name = request.POST.get("name")
        camera.camera_source = request.POST.get("camera_source")
        try:
            camera.threshold = float(request.POST.get("threshold", 0.6))
        except ValueError:
            pass 
        
        camera.save()
        messages.success(request, "Camera updated successfully!")
        return redirect("camera_config_list")
        
    return render(request, "camera_config_update.html", {"camera": camera})

def camera_config_delete(request, pk):
    camera = CameraConfiguration.objects.filter(pk=pk).first()
    if camera:
        camera.delete()
        messages.success(request, "Camera deleted successfully!")
    return redirect("camera_config_list")