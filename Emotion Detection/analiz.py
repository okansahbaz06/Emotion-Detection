#-*- coding: utf-8 -*-
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

yüz_sınıflandırıcı = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')#Yüz tespiti yapan xml dosyası
sınıflandırıcı = load_model('./Emotion_Detection.h5')#Eğitilmiş model

duygular = ['Kizgin','Mutlu','Dogal','Uzgun','Saskin']#Duygular bir dizi içerisinde tutuluyor.

cap = cv2.VideoCapture(1)#Kameranın açılıp anlık görüntü tespiti yapması



while True:

    ret, frame = cap.read()#tek bir görüntü yakalanıp kontrol ediliyor.
    gri = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    yüzler = yüz_sınıflandırıcı.detectMultiScale(gri,1.3,5)

    for (x,y,w,h) in yüzler:#yüzün konumlarını bulup fonksiyona yolluyor.
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#Yüzü dikdörtgen içine alıp konumlandırıyoruz.
        gri_yüz = gri[y:y+h,x:x+w]#Tanıması gereken yer olan yüz için yazılan özel kod
        gri_yüz = cv2.resize(gri_yüz,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([gri_yüz])!=0:#Görüntüyü boyutlandırdıktan sonra diziye dönüştürüyoruz.
            yüz = gri_yüz.astype('float')/255.0
            yüz = img_to_array(yüz)
            yüz = np.expand_dims(yüz,axis=0)

            # Burada duygular tahmin ediliyor.Her duygunun olasılığı tahmin ediliyor.
            tahmin = sınıflandırıcı.predict(yüz)[0]
            etiket=duygular[tahmin.argmax()]#Burda duygunun ismine erişiyoruz.
            etk_pst = (x,y)
            cv2.putText(frame,etiket,etk_pst,cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,255,0),3)#Kafanın üzerine yazdırma
        else:
            cv2.putText(frame,'Yuz bulunamadi',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Duygu Analizi',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):#q tuşu ile kamerayı kapat
        break

cap.release()
cv2.destroyAllWindows()


























