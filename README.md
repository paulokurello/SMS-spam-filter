# SMS-spam-filter

Projekt realizuje filtr SPAM wiadomości SMS w pythonie przy wykorzystaniu Naive Bayes.
Baza wiadomości SMS wykorzystana przy realizacji: https://www.kaggle.com/uciml/sms-spam-collection-dataset.

Projekt implementuje 2 moduły wyliczania prawdopodobieństw:
 - Bag of Words
 - TF-IDF
 oraz zapewnia preprocessing wiadomości:
  - podstawowy (usuwanie znaków interpunkcyjnych oraz podział wiadomości na słowa)
  - zaawansowany (aktywowany przy pomocy specjalnych flag)
  
  Na preprocessing zaawansowany składają się:
  - zamiana wszystkich liter na małe (wybierane za pomocą wartości TRUE flagi --lower)
  - usunięcie wyrazów kluczowych języka angielskiego, takich jak "the" (wybierane za pomocą wartości TRUE flagi --rmStop)
  - redukcja wyrazów do ich trzonów, bez końcówek fleksyjnych: "goes -> "go" (wybierane za pomocą wartości TRUE flagi --stem)
  
  Moduł testujący skuteczność uczenia dokonuje tego poprzez wyliczenie 4 wartości:
  - precision
  - recall
  - F1-score
  - accuracy
  
  opisanych dokładnie tutaj: https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/?fbclid=IwAR0kn1i-WPS8zRvhhfHf93OApqYJNnUYXKHI3MbzPzyJHgpdd4Xo6VrkTts
  
  Projekt wymaga zainstalowanych:
  -numpy
  -pandas
  -nltk
  
  Autorzy:
  - Małgorzata Górecka
  - Paweł Kurowski
