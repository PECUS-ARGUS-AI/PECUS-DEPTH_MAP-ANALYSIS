from ultralytics import YOLO
import cv2
import numpy as np
from coordenadas_depth_map import generate_depth_and_pointcloud, get_3d_point

model = YOLO('best.pt')

img = cv2.imread('teste_gado.jpg')

"""h, w, _ = img.shape
crop = img[0:h, 0:int(w * 0.8)]
print(crop.shape)"""


results = model.predict(source=img, imgsz=544, conf=0.70, save=True)

points = generate_depth_and_pointcloud(
    image=img,
    width=img.shape[1],
    height=img.shape[0],
    save_dir="saida_2"
)

width_crop = img.shape[1]  

comprimento_total = 0
larguras = []
alturas = []

for r in results:
    img_contorno = r.orig_img.copy()

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        print(f"Caixa delimitadora: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        cv2.rectangle(img_contorno, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(img_contorno, (x1, y1), 6, (0, 0, 255), -1)
        cv2.circle(img_contorno, (x2, y1), 6, (0, 0, 255), -1)
        cv2.circle(img_contorno, (x1, y2), 6, (0, 0, 255), -1)
        cv2.circle(img_contorno, (x2, y2), 6, (0, 0, 255), -1)
        """x1y1 = get_3d_point(points, width_crop, x1, y1)
        x2y1 = get_3d_point(points, width_crop, x2, y1)
        x2y2 = get_3d_point(points, width_crop, x2, y2)
        base = np.linalg.norm(x1y1 - x2y1)
        height = np.linalg.norm(x2y1 - x2y2)
        area = base * height"""

        pt_left_3d = get_3d_point(points, width_crop, x1, y1)
        pt_right_3d = get_3d_point(points, width_crop, x2, y1)
        pt_top_center_3d = get_3d_point(points, width_crop, (x1 + x2)//2, y1)
        pt_bottom_center_3d = get_3d_point(points, width_crop, (x1 + x2)//2, y2)

        comprimento_total = np.linalg.norm(pt_right_3d - pt_left_3d)
        area = comprimento_total * np.linalg.norm(pt_bottom_center_3d - pt_top_center_3d)


    if r.masks is not None:
        for seg in r.masks.xy:
            pontos = np.array(seg, dtype=np.int32)

            idx_esq = np.argmin(pontos[:, 0])
            idx_dir = np.argmax(pontos[:, 0])

            p_esq = tuple(pontos[idx_esq])
            p_dir = tuple(pontos[idx_dir])

            x_esq, y_esq = int(p_esq[0]), int(p_esq[1])
            x_dir, y_dir = int(p_dir[0]), int(p_dir[1])

            print(f"Ponto extremo esquerdo (x_min): ({x_esq}, {y_esq})")
            print(f"Ponto extremo direito (x_max): ({x_dir}, {y_dir})")
            print("-" * 60)
            cv2.circle(img_contorno, p_esq, 8, (0, 0, 255), -1)
            cv2.circle(img_contorno, p_dir, 8, (255, 0, 0), -1)
            cv2.line(img_contorno, p_esq, p_dir, (255, 255, 0), 3)

            #pt_left_3d = get_3d_point(points, width_crop, x_esq, y_esq)
            #pt_right_3d = get_3d_point(points, width_crop, x_dir, y_dir)

            mask = np.zeros(img_contorno.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pontos], 255)

            num_pontos = 10
            xs = np.linspace(x_esq, x_dir, num_pontos)

            for x in xs:
                x = int(x)
                coluna = mask[:, x]
                ys = np.where(coluna > 0)[0]

                if len(ys) > 0:
                    y_top, y_bottom = ys[0], ys[-1]

                    cv2.line(img_contorno, (x, y_top), (x, y_bottom), (0, 255, 255), 2)
                    cv2.circle(img_contorno, (x, y_top), 4, (0, 0, 255), -1)
                    cv2.circle(img_contorno, (x, y_bottom), 4, (255, 0, 0), -1)
                    
                    y_meio = int((y_top + y_bottom) / 2)
                    cv2.circle(img_contorno, (x, y_meio), 5, (0, 255, 0), -1)

                    print(f"Ponto top px: {x, int(y_top)}")
                    print(f"Ponto bottom px: {x, int(y_bottom)}")
                    print(f"Ponto meio px: {x, int(y_meio)}")
                    pt_top_3d = get_3d_point(points, width_crop, x, y_top)
                    pt_bottom_3d = get_3d_point(points, width_crop, x, y_bottom)
                    pt_middle_3d = get_3d_point(points, width_crop, x, y_meio)
                    print(f"Ponto top px: {pt_top_3d}")
                    print(f"Ponto bottom px: {pt_bottom_3d}")
                    print(f"Ponto meio px: {pt_middle_3d}")
                    largura = np.linalg.norm(pt_top_3d - pt_bottom_3d)
                    print(f"Largura {largura}")
                    larguras.append(largura)
                    altura = abs(pt_middle_3d[2])
                    print(f"Altura {altura}")
                    alturas.append(altura)
                    print("-"*60)

    media_altura = np.mean(alturas)
    print(f"Media Altura {media_altura}")
    #soma_largura = float(np.sum(larguras))
    media_largura = np.mean(larguras)
    #largura_final = soma_largura - area
    #print(f"Soma Largura - Area {soma_largura} | {area}")
    print(f"Largura final {media_largura}")
    print(f"Comprimento: {comprimento_total}")

    volume = comprimento_total * media_largura * media_altura

    print(f"Volume: {volume/1000000:.2f}")
    print(f"Massa: {volume*1017:.2f}")

    cv2.imshow("Bovino", img_contorno)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
