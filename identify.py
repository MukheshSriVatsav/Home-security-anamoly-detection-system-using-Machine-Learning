from openpyxl import Workbook
def identify():
    cap = cv2.VideoCapture(0)

    filename = "haarcascade_frontalface_default.xml"

    paths = [os.path.join("persons", im) for im in os.listdir("persons")]
    labelslist = {}
    for path in paths:
        labelslist[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]

    print(labelslist)
    recog = cv2.face.LBPHFaceRecognizer_create()

    recog.read('model.yml')

    cascade = cv2.CascadeClassifier(filename)

    # Create a new Excel workbook
    wb = Workbook()
    # Create a worksheet in the workbook
    ws = wb.active
    # Add column headers
    ws.append(["ID", "Name"])

    while True:
        _, frm = cap.read()

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.3, 2)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]

            label = recog.predict(roi)

            if label[1] < 100:
                identified_name = labelslist[str(label[0])]
                cv2.putText(frm, f"{identified_name} + {int(label[1])}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # Write the identified name to Excel
                ws.append([str(label[0]), identified_name])
            else:
                cv2.putText(frm, "unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("identify", frm)

        if cv2.waitKey(1) == 27:
            # Save the Excel file
            wb.save("identified_names.xlsx")
            cv2.destroyAllWindows()
            cap.release()
            break
