import requests
import json
from elasticsearch import Elasticsearch

def json_load(path, mode='r'):
    with open(path, mode) as reader:
        contents = json.load(reader)        
    return contents


text = '''אם נבחר לדבר על נושא הטכנולוגיה וההתפתחות שלה בעידן המודרני, יש לשים לב לכמה מהמישורים המרכזיים המשפיעים על חיי האדם והחברה. כיום, טכנולוגיה משפיעה על כמעט כל נקודת חיינו, מהמוצרים שאנו צורכים ועד לאופן שבו אנו תואמים עם העולם סביבנו.
במאה ה-21, הטכנולוגיה התקדמה בצער והופכת לחלק בלתי נפרד מהיומיום שלנו. חידושים כמו חיבור אינטרנט אלחוטי, טלפוניה ניידת, ושירותי ענן משנים את הפנייה שלנו אל העולם. כלי תקשורת אלקטרוניים הופכים לכלי יומיומיים שבעזרתם אנו יכולים ליצור קשר עם אנשים בכל רגע נתון, בכל נקודה בעולם.
אחת מההתפתחויות המרכזיות בעידן המודרני היא הפריצה הדיגיטלית, שמציינת את המעבר מהחברה התעשייתית לחברה המבוססת על ידע ומידע. בעידן הדיגיטל, מידע הוא יכולת עיקרית, והיכולת לנהל ולעבד מידע מהווה יתרון אדיר. זה משפיע על דרכים רבות, כמו למשל בתחום המחקר הרפואי, שבו ניתן לנתח מידע רפואי ולפתח תרומות חדשות במהירות וביעילות גבוהה.
בנוסף, טכנולוגיה משפיעה על אופן העבודה והתקשורת בעסקים. עם התפשטות המחשבים והרשתות, העבודה מתבצעת פעמים רבות גם מחוץ למשרדים הפיזיים, והיכולת לשתף פעולה ולתקשר מרחוק הופכת לזכות בחשיבות מרבית.
אולם, יש גם צדדים שליליים להתפתחות טכנולוגית. דמיון חברתי נוזלי, חשיבה בקפיצים, ובעיקר התקררות יחסים אנושיים עשויים להיות התוצאה של ניכור התקשורת האנושית עקב התמקדות יתר על הטכנולוגיה.
בסיכום, יש להבין כי הטכנולוגיה מצויה בלב החיים המודרניים, והשפעתה משפיעה על רבים מיבני החיים שלנו, מהקשרים אישיים ועד לסגנונות העבודה והלמידה. האתגר הוא למצוא את האיזון הנכון ולהשתמש בטכנולוגיה לתועלת האדם ולשיפור איכות החיים.
'''

mappings = json_load('mappings_he.json')
settings = json_load('settings_he.json')

try:
    es = Elasticsearch("http://localhost:9200", request_timeout=1000)
    print(es)
except Exception as e:
    print(e)

url = "http://localhost:9200/_analyze?pretty"

headers = {
    "Content-Type": "application/json"
}

data = {
    "text": text,
    "tokenizer": "whitespace",
    "filter": ["heb_lemmas", "heb_stopwords"]
}

response = requests.get(url, headers=headers, json=data)
print(response.text)

# Create an index
if es.indices.exists(index="test_index"):
    response = es.indices.delete(index="test_index")
    print(response)
response = es.indices.create(index="test_index", settings=settings, mappings=mappings)
print(response)

document = {
    "text":  text,
}

# Indexing the document
response = es.index(index="test_index", document=document)
print(response)
