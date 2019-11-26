from gtts import gTTS
import os
import nltk
from newspaper import Article
article=Article("https://timesofindia.indiatimes.com/city/delhi/massive-protest-by-students-on-jnu-campus/articleshow/72002390.cms")
article.download()
article.parse()
nltk.download('punkt')
article.nlp()
mytext=article.text
language='en'
myobj=gTTS(text=mytext,lang=language,slow=False)
myobj.save("read_articlea1.mp3")
os.system("start read_articlea1.mp3")
 
"""file=open("aa.txt")
file_name=file.read()
language='en'
myobj=gTTS(text=file_name,lang=language,slow=False)
myobj.save("welcome.mp3")
os.system("start welcome.mp3")
"""

