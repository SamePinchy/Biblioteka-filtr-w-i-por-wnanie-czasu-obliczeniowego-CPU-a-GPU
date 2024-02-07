# Biblioteka filtrów i porównanie czasu obliczeniowego CPU a GPU

## Opis Projektu
Program oparty na CUDA wykorzystujący akcelerację GPU do nakładania różnorodnych filtrów na obrazy. Program zapewnia opcje wyświetlania przefiltrowanych obrazów za pomocą biblioteki OpenCV oraz umożliwia porównywanie czasu obliczeniowego między CPU a GPU (za pomocą biblioteki chrono dla CPU oraz cudaEvent dla GPU) dla każdego filtra.
Zaprojektowane filtry:
- Filtr Sobela: Wykrywa krawędzie na obrazach za pomocą operatora Sobela.
- Filtr Laplace'a: Wzmacnia krawędzie i detale na obrazach za pomocą operatora Laplace'a.
- Filtr Kuwahara: Stosuje filtr Kuwahara, tworząc efekt malarski.
- Filtr Piramidalny: Wygładza obrazy za pomocą filtru piramidalnego.
- Filtr Mozaikowy: Generuje efekt mozaiki poprzez pikselizację obrazu.
- Filtr Wypukły 3D: Dodaje efekt wypukłości 3D do obrazów.
- Filtr Sepii: Stosuje efekt sepia do obrazów, nadając im nostalgiczny charakter.

## Przykłady Działania

Wykresy przedstawiają graficzne porównanie czasu obliczeniowego CPU oraz GPU
- kolor czerwony -> CPU
- kolor czarny -> GPU

### Zdjęcie pierwotne

![czlek](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/3e4cb4a9-9d27-4e1f-b40d-aa1db383f44b)

### Filtr Sobela:

![1_1](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/2eb4a6b5-d835-499a-ac1c-a4ab8801de9e)
![1_2](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/695c3eee-19c3-448c-8bcb-031e19870cd4)
- Czas wykonania na CPU: 3730 mikrosekundy
- Czas wykonania na GPU: 867.808 mikrosekundy

### Filtr Laplace'a:

![2_1](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/b31259cf-102a-4b8f-8011-08e48ff2f26e)
![2_2](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/c9834e34-6dee-4064-9f8a-5e8fbecc1e09)
- Czas wykonania na CPU: 1483 mikrosekundy
- Czas wykonania na GPU: 802.4 mikrosekundy

### Kuwahara:

![3_1](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/1065d11a-a433-403a-ba56-cb2833f2766b)
![3_2](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/7883ca0c-c4ab-4ac7-ae80-c3af20f0251f)
- Czas wykonania na CPU: 30524 mikrosekundy
- Czas wykonania na GPU: 2353.79 mikrosekundy

### Piramidalny:

![4_1](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/4413c965-623e-43f0-99e8-f46a51632a6b)
![4_2](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/0c4042d4-f504-4a66-876a-24a25af8e6d5)
- Czas wykonania na CPU: 16121 mikrosekundy
- Czas wykonania na GPU: 7810.21 mikrosekundy

### Mozaikowy:

![5_1](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/16a82f0a-2e37-4076-9289-6a9ca9c21c45)
![5_2](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/032a8852-4420-4068-922f-1a327f44176f)
- Czas wykonania na CPU: 15653 mikrosekundy
- Czas wykonania na GPU: 4208.96 mikrosekundy6

### Wypukły 3D:

![6_1](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/f79748d6-9a27-4b25-8fe2-a87487e2770e)
![6_2](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/a6d8818e-1587-49f6-a38d-bfdf6918b76a)
- Czas wykonania na CPU: 20207 mikrosekundy
- Czas wykonania na GPU: 1445.44 mikrosekundy

### Sepii:

![7_1](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/11bc2ae0-e0e8-4cfd-bc02-2f4b8af0f2cc)
![7_2](https://github.com/SamePinchy/Biblioteka-filtr-w-i-por-wnanie-czasu-obliczeniowego-CPU-a-GPU/assets/106782201/6d97371d-5fae-4850-af48-6c2d72db6376)
- Czas wykonania na CPU: 17799 mikrosekundy
- Czas wykonania na GPU: 12602.4 mikrosekundy

## Autorzy
- Miłosz Smolarczyk 189008
- Jakub Kabat 191339
