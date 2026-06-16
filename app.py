import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def check_gesture(lms):
    # More accurate detection using joints (DIP joints)
    index_open = lms[8].y < lms[6].y
    middle_open = lms[12].y < lms[10].y
    ring_open = lms[16].y < lms[14].y
    pinky_open = lms[20].y < lms[18].y
    
    if not index_open and not middle_open and not ring_open and not pinky_open:
        return "Rock"
    elif index_open and middle_open and ring_open and pinky_open:
        return "Paper"
    elif index_open and middle_open and not ring_open and not pinky_open:
        return "Scissors"
    return "Unknown"

def check_winner(u, c):
    if u == c: return "Draw!"
    if (u=="Rock" and c=="Scissors") or (u=="Paper" and c=="Rock") or (u=="Scissors" and c=="Paper"):
        return "You Win!"
    return "Computer Wins!"

cap = cv2.VideoCapture(0)
# Setup Video Recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('game_demo.avi', fourcc, 20.0, (640, 480))

last_gesture = None
typing_text = ""
typing_counter = 0

print("Game Started! Show your hand. Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 1. Logic for Gesture Detection
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            u = check_gesture(hand_lms.landmark)
            
            # Logic: Only calculate result if gesture changed and is valid
            if u != "Unknown" and u != last_gesture:
                c = random.choice(["Rock", "Paper", "Scissors"])
                res = check_winner(u, c)
                typing_text = f"You: {u} | Comp: {c} | {res}"
                typing_counter = 0
                last_gesture = u
    
    # 2. Typewriter Effect Logic
    if typing_text:
        if typing_counter < len(typing_text) * 4:
            display_text = typing_text[:typing_counter // 4]
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            typing_counter += 1
            
    out.write(frame)
    cv2.imshow("Hand Game", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()