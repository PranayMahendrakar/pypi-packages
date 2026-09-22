"""The evidence tables behind :func:`multilingual_text.detect`.

Three kinds of evidence live here:

``STOPWORDS``
    The most frequent function words of each space-separated language.  A word
    that several languages share counts for less, and the weights are worked
    out once at import time so no list has to be curated against the others.

``MARKERS``
    Short character sequences for the languages that do not put spaces between
    words (Chinese, Japanese, Korean, Thai).  These are matched as substrings.

``ALLOWED_MARKS`` / ``DISTINCTIVE``
    Which accented Latin letters an orthography does and does not use.  This is
    what separates Spanish from Portuguese, or German from Dutch, when the
    function words alone are not enough.

Nothing here is downloaded and nothing is trained: the tables are plain data.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

__all__ = [
    "ALLOWED_MARKS",
    "CANDIDATES",
    "DISTINCTIVE",
    "LANGUAGE_NAMES",
    "LANGUAGE_SCRIPT",
    "MARKERS",
    "MARKER_WEIGHTS",
    "STOPWORDS",
    "SUPPORTED",
    "WORD_WEIGHTS",
]


def _words(blob: str) -> FrozenSet[str]:
    return frozenset(blob.split())


STOPWORDS: Dict[str, FrozenSet[str]] = {
    "en": _words(
        "the be to of and a in that have it for not on with he as you do at this"
        " but his by from they we say her she or an will my one all would there"
        " their what is are was were has been its if about which when who how i"
        " me your can more no out up so than then them these some into only"
        " other new could time also been being am very much many make know take"
        " see come think look want give use find tell ask work because any day"
        " most us hello world thanks please yes"
        " one two three after before while where why still even just back like"
        " under through between both each own same such too way well life man"
        " woman year good great little first last long here there now today"
        " yesterday tomorrow everyone nobody something nothing which whose"
        " during against without"
    ),
    "es": _words(
        "el la de que y a en un ser se no haber por con su para como estar tener"
        " le lo todo pero mas hacer o poder decir este ir otro ese si me ya ver"
        " porque dar cuando muy sin vez mucho saber sobre mi mismo yo tambien"
        " hasta dos querer entre asi primero desde grande eso ni nos llegar"
        " tiempo ella dia uno bien poco deber entonces cosa tanto hombre nuestro"
        " tan donde ahora parte despues vida siempre creer hablar nada cada"
        " seguir menos nuevo los las una unos unas del al es son esta estan era"
        " hola gracias buenos"
        " bajo contra durante segun hacia aunque mientras ademas tampoco nunca"
        " aqui alli ayer manana hoy todos todas algo nadie alguien cual cuanto"
        " quien esto esa esas esos aquel sino solo pues"
    ),
    "fr": _words(
        "le de un etre et a il avoir ne je son que se qui ce dans en du elle au"
        " pour pas vous par sur faire plus dire me on mon lui nous comme mais"
        " pouvoir avec tout y aller voir bien ou sans tu leur homme si deux moi"
        " vouloir te femme venir quand grand celui la les des une est sont etait"
        " aux ces cette ses nos vos tres aussi alors chaque toujours jamais deja"
        " apres avant encore entre donc ainsi cest nest jai bonjour merci"
        " dessus dessous chez contre pendant depuis selon vers meme autre peu"
        " beaucoup trop assez ici oui non quoi comment pourquoi combien quel"
        " quelle cela ceci etes sommes ont avons avez fait dit peut doit faut"
        " rien tous toute toutes pres loin jour temps chose monde etaient sera"
        " ceux celles"
    ),
    "de": _words(
        "der die und in den von zu das mit sich des auf fur ist im dem nicht ein"
        " eine als auch es an werden aus er hat dass sie nach wird bei einer um"
        " am sind noch wie einem uber einen so zum war haben nur oder aber vor"
        " zur bis mehr durch man sein wurde sei ich wir ihr ihre diese dieser"
        " kann wenn schon dann alle was wo sehr keine wieder immer machen sagen"
        " gehen kommen hallo danke bitte guten"
        " dieses jener jede jeden alles etwas nichts jemand niemand welche"
        " warum wann weil obwohl damit ohne gegen zwischen wahrend seit heute"
        " morgen gestern nie oft hier dort gut neu gross klein tag zeit mensch"
        " unter denn doch also jetzt schnell"
    ),
    "it": _words(
        "di e il la che un a per in non una con sono mi si ho ma lo ha le come io"
        " questo hai del sei gli della da al nel piu anche se cosa quando tutto"
        " essere molto perche ci dei alla dove chi tu noi voi loro suo mio tuo"
        " delle degli nella sulla ogni ancora gia cosi senza dopo prima tra fra"
        " sulla questa quello quella fare dire ciao grazie buongiorno"
        " sopra sotto poi quindi pero sempre oggi bene tutti tutte alcuni"
        " nostro vostro lei lui erano sara hanno abbiamo siamo stato dalla dai"
        " nei sui col negli agli alle giorno tempo cose mondo qui qua niente"
        " nulla oppure mentre infatti verso contro senza"
    ),
    "pt": _words(
        "de a o que e do da em um para com nao uma os no se na por mais as dos"
        " como mas foi ao ele das tem seu sua ou ser quando muito ha nos ja esta"
        " eu tambem so pelo pela ate isso ela entre era depois sem mesmo aos ter"
        " seus quem nas me esse eles estao voce tinha foram essa num nem suas"
        " meu minha tem numa elas havia seja qual lhe deles este dele voces"
        " sobre obrigado ola bom"
        " sob contra durante segundo embora enquanto alem nunca sempre aqui"
        " ali ontem amanha hoje todos todas algo ninguem alguem quanto isto"
        " essas esses aquele apenas porque assim ainda"
    ),
    "nl": _words(
        "en van ik te dat die in een hij het niet zijn is was op aan met als voor"
        " had er maar om hem dan zou of wat mijn dit zo door over ze zich bij ook"
        " tot je mij uit daar haar naar heb hoe heeft hebben deze want nog zal me"
        " zij nu geen omdat iets worden toch al waren veel meer doen toen moet"
        " ben zonder kan hun dus alles onder eens hier wie werd altijd wordt"
        " kunnen ons zelf tegen na wil kon niets uw iemand geweest andere hallo"
        " bedankt goedemorgen"
        " de den gaat gaan komen zien weten maken moeten wel waar zoals echt"
        " even weer samen tussen tijdens binnen buiten huis dag tijd mens"
        " groot klein goed nieuw elke alle iedereen niemand welke hoeveel"
        " waarom wanneer omhoog terug dus"
    ),
    "pl": _words(
        "w i z na do sie nie to ze jest o a jak od po za ale co tak ja ty on ona"
        " my wy oni byc miec ktory ktora ktore przez dla przy pod nad juz tylko"
        " jeszcze bardzo gdzie kiedy dlaczego wszystko moze mozna trzeba jego jej"
        " ich nasz wasz ten ta te tego tym jeden dwa czy lub oraz wiec tez tam"
        " tutaj dobrze teraz czesc dziekuje prosze"
        " sa byl byla bylo mam masz ma jestem jestes jesli zeby albo ani znowu"
        " zawsze nigdy dzisiaj wczoraj jutro wszyscy nikt ktos cos nic ile nad"
        " pod bez przed razem potem bardziej"
    ),
    "sv": _words(
        "och i att det som en pa ar av for med till den har de inte om ett han"
        " men var jag sig fran vi sa kan man nar vid da du hon eller ska skulle"
        " efter alla andra over nagon vara sin sitt har dar nu mycket bara hur"
        " vad mer ut upp ner ocksa vill far gor blir hade hej tack god"
        " sedan igen redan kanske alltid aldrig hela varje mellan under genom"
        " utan mot hos detta dessa dess sina deras mig dig oss honom henne dem"
        " vem vilken varfor eftersom medan samt nog lite stor liten bra ny dag"
        " tid manniska mycket"
    ),
    "tr": _words(
        "bir bu ve icin ile da de ne ben sen o biz siz onlar var yok cok daha en"
        " gibi kadar sonra once ama veya ya her hic sey olarak olan oldu degil mi"
        " ki ise boyle soyle nasil neden nicin kim hangi nerede zaman simdi"
        " bugun yine sadece hem ancak cunku eger gore uzere beni seni onu bizi"
        " sizi bana sana ona merhaba tesekkur gunaydin"
        " bile iste yani artik hala henuz asla hep bazen burada orada dun"
        " yarin herkes kimse kac nasil yapmak olmak buyuk kucuk iyi kotu yeni"
        " eski gun"
    ),
    "vi": _words(
        "va cua la co duoc trong cho khong nguoi mot nhung voi cac de da nay khi"
        " den tu ra thi se toi ban anh chi em no ho chung ma nhu ve nen cung con"
        " rat nhieu neu vi do ai gi sao nao hay hoac di lam noi biet thay xin"
        " chao cam on"
        " tren duoi ngoai sau truoc roi dang van moi qua lai len xuong kia day"
        " nua rang cho chua tat ca minh"
    ),
    "id": _words(
        "yang dan di itu dengan untuk tidak ini dari dalam akan pada juga saya ke"
        " karena tersebut bisa ada mereka lebih kata sudah atau saat oleh adalah"
        " harus kami kita anda dia orang bahwa tahun hari banyak sangat masih"
        " belum bukan apa siapa bagaimana kapan dimana agar tetapi namun sebagai"
        " seperti terima kasih selamat halo"
        " atas bawah setelah sebelum ketika sambil lalu kemudian sekarang"
        " kemarin besok semua berapa mana antara tanpa setiap setiap sendiri"
        " begitu jadi"
    ),
    "ru": _words(
        "и в не на я что он с а как это но они мы вы к из у за от до по для при"
        " же бы ли да нет был была было были есть быть его её их мой твой наш ваш"
        " свой весь все всё тот этот эта эти там тут здесь где когда потому если"
        " чтобы очень ещё уже только также может надо нужно человек время год"
        " день так или тебя меня себя кто какой очень привет спасибо пожалуйста"
        " здравствуйте мир"
    ),
    "uk": _words(
        "і в не на я що він з а як це але вони ми ви до із у за від по для при же"
        " би чи так ні був була було були є бути його її їх мій твій наш ваш свій"
        " весь всі все той цей ця ці там тут де коли тому якщо щоб дуже ще вже"
        " тільки також може треба людина час рік день та або хто який яка мене"
        " тебе себе привіт дякую будь ласка світ"
    ),
    "el": _words(
        "και το του της τα των στο στη στον με για από που δεν να θα είναι ήταν"
        " ένα μια ένας αυτό αυτή αυτός εγώ εσύ εμείς εσείς αλλά ή αν όταν γιατί"
        " πώς τι ποιος όλα πολύ ακόμα ήδη μόνο επίσης στην την τον οι η ο σε ως"
        " γεια σας ευχαριστώ κόσμε"
    ),
    "ar": _words(
        "في من على أن إلى عن مع هذا هذه ذلك التي الذي كان كانت لا ما هو هي هم نحن"
        " أنا أنت قد كل بعد قبل بين عند حتى أو ثم لكن إذا كما أيضا الآن اليوم سنة"
        " يوم ولا وهو وهي إن لم لن هناك كيف لماذا أين متى مرحبا شكرا السلام عليكم"
        " العالم"
    ),
    "he": _words(
        "את של על לא אני זה הוא היא אנחנו אתם הם עם כל אבל גם יש אין מה מי איך"
        " למה כי אם או היה הייתה אחרי לפני בין עוד כבר רק מאוד היום שנה יום כאן"
        " שם אנו הזה הזאת שלי שלך שלנו שלום תודה עולם"
    ),
    "hi": _words(
        "है और का की के में को से पर यह वह हैं था थी थे नहीं कि एक हो भी लिए कर"
        " गया तो ही या अगर जब क्या कौन कैसे कहाँ बहुत अब आज साल दिन मैं तुम आप हम"
        " वे उन इस उस लेकिन जो सब कुछ अपने बाद होता करने वाले नमस्ते धन्यवाद"
        " दुनिया"
    ),
    "mr": _words(
        "आहे आणि च्या ची चे मध्ये ला ने वर हे तो ती ते आहेत होता होती होते नाही"
        " की एक पण ही साठी केले झाले तर किंवा जर जेव्हा काय कोण कसे कुठे खूप आता"
        " आज वर्ष दिवस मी तू तुम्ही आम्ही त्या या असे यांनी करून सर्व नमस्कार"
        " धन्यवाद जग"
    ),
    "bn": _words(
        "এবং এই সেই তার আমি তুমি আপনি আমরা তারা হয় ছিল না কি যে করে থেকে জন্য"
        " সঙ্গে উপর মধ্যে একটি কিন্তু বা যদি কখন কেন কোথায় খুব এখন আজ বছর দিন"
        " করা হবে আছে তিনি নিয়ে সব নমস্কার ধন্যবাদ পৃথিবী"
    ),
    "ta": _words(
        "மற்றும் இந்த அந்த என்று ஒரு இல்லை நான் நீ நீங்கள் நாங்கள் அவர் அவள்"
        " அவர்கள் இது அது ஆனால் அல்லது என்ன யார் எப்படி எங்கே மிகவும் இப்போது"
        " இன்று ஆண்டு நாள் உள்ளது இருந்து மேலும் செய்ய வேண்டும் வணக்கம் நன்றி"
        " உலகம்"
    ),
    "te": _words(
        "మరియు ఈ ఆ ఒక కాదు నేను నువ్వు మీరు మేము అతను ఆమె వారు ఇది అది కానీ లేదా"
        " ఏమి ఎవరు ఎలా ఎక్కడ చాలా ఇప్పుడు ఈరోజు సంవత్సరం రోజు ఉంది నుండి కోసం తో"
        " చేయడం అన్ని నమస్కారం ధన్యవాదాలు ప్రపంచం"
    ),
    "gu": _words(
        "અને આ તે એક નથી હું તું તમે અમે તેઓ છે હતું પણ અથવા શું કોણ કેવી ક્યાં"
        " ખૂબ હવે આજે વર્ષ દિવસ માટે થી સાથે માં ના ની નું કરવા બધા નમસ્તે આભાર"
        " દુનિયા"
    ),
    "pa": _words(
        "ਅਤੇ ਇਹ ਉਹ ਇੱਕ ਨਹੀਂ ਮੈਂ ਤੂੰ ਤੁਸੀਂ ਅਸੀਂ ਉਹਨਾਂ ਹੈ ਸੀ ਪਰ ਜਾਂ ਕੀ ਕੌਣ ਕਿਵੇਂ"
        " ਕਿੱਥੇ ਬਹੁਤ ਹੁਣ ਅੱਜ ਸਾਲ ਦਿਨ ਲਈ ਤੋਂ ਨਾਲ ਵਿੱਚ ਦਾ ਦੀ ਦੇ ਕਰਨ ਸਭ ਸਤ ਸ੍ਰੀ"
        " ਅਕਾਲ ਧੰਨਵਾਦ ਦੁਨੀਆ"
    ),
    "kn": _words(
        "ಮತ್ತು ಈ ಆ ಒಂದು ಅಲ್ಲ ನಾನು ನೀನು ನೀವು ನಾವು ಅವನು ಅವಳು ಅವರು ಇದು ಅದು ಆದರೆ"
        " ಅಥವಾ ಏನು ಯಾರು ಹೇಗೆ ಎಲ್ಲಿ ತುಂಬಾ ಈಗ ಇಂದು ವರ್ಷ ದಿನ ಇದೆ ಇಂದ ಗಾಗಿ ಜೊತೆ"
        " ಮಾಡಲು ಎಲ್ಲಾ ನಮಸ್ಕಾರ ಧನ್ಯವಾದ ಪ್ರಪಂಚ"
    ),
    "ml": _words(
        "ഒരു ഈ ആ അല്ല ഞാൻ നീ നിങ്ങൾ ഞങ്ങൾ അവൻ അവൾ അവർ ഇത് അത് പക്ഷേ"
        " അല്ലെങ്കിൽ എന്ത് ആര് എങ്ങനെ എവിടെ വളരെ ഇപ്പോൾ ഇന്ന് വർഷം ദിവസം ഉണ്ട്"
        " നിന്ന് വേണ്ടി കൂടെ ചെയ്യാൻ എല്ലാ നമസ്കാരം നന്ദി ലോകം"
    ),
}

# Languages written without spaces: matched as substrings, not as words.
MARKERS: Dict[str, Tuple[str, ...]] = {
    "zh": (
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一个",
        "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看",
        "好", "自己", "这", "那", "他", "她", "们", "中国", "可以", "什么",
        "因为", "所以", "但是", "如果", "我们", "他们", "现在", "时候", "知道",
        "世界", "你好", "谢谢",
    ),
    "ja": (
        "の", "に", "は", "を", "た", "が", "で", "て", "と", "し", "れ", "さ",
        "ある", "いる", "も", "する", "から", "な", "こと", "として", "です",
        "ます", "ました", "ない", "これ", "それ", "あれ", "私", "日本", "って",
        "けど", "でも", "そして", "ください", "ありがとう", "こんにちは",
        "世界",
    ),
    "ko": (
        "이", "그", "저", "것", "수", "등", "및", "을", "를", "은", "는", "에",
        "에서", "으로", "와", "과", "하다", "있다", "없다", "합니다", "입니다",
        "하지만", "그리고", "우리", "나는", "저는", "안녕하세요", "감사합니다",
        "세계", "한국",
    ),
    "th": (
        "และ", "ที่", "ใน", "ของ", "เป็น", "ไม่", "มี", "ได้", "จะ", "การ",
        "ความ", "กับ", "ให้", "ไป", "มา", "นี้", "นั้น", "ผม", "ฉัน", "คุณ",
        "เรา", "เขา", "แต่", "หรือ", "ถ้า", "เมื่อ", "อะไร", "ใคร", "มาก",
        "ตอนนี้", "วันนี้", "ปี", "วัน", "สวัสดี", "ขอบคุณ", "โลก",
    ),
}

#: Which script each supported language is written in.
LANGUAGE_SCRIPT: Dict[str, str] = {
    "en": "Latin", "es": "Latin", "fr": "Latin", "de": "Latin", "it": "Latin",
    "pt": "Latin", "nl": "Latin", "pl": "Latin", "sv": "Latin", "tr": "Latin",
    "vi": "Latin", "id": "Latin",
    "ru": "Cyrillic", "uk": "Cyrillic",
    "el": "Greek", "ar": "Arabic", "he": "Hebrew",
    "hi": "Devanagari", "mr": "Devanagari",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "gu": "Gujarati",
    "pa": "Gurmukhi", "kn": "Kannada", "ml": "Malayalam",
    "zh": "Han", "ja": "Hiragana", "ko": "Hangul", "th": "Thai",
}

#: Which languages a given dominant script puts on the table.  Order is the
#: tie-break when the evidence is a dead heat.
CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "Latin": ("en", "es", "fr", "de", "it", "pt", "nl", "pl", "sv", "tr", "vi", "id"),
    "Cyrillic": ("ru", "uk"),
    "Greek": ("el",),
    "Arabic": ("ar",),
    "Hebrew": ("he",),
    "Devanagari": ("hi", "mr"),
    "Bengali": ("bn",),
    "Tamil": ("ta",),
    "Telugu": ("te",),
    "Gujarati": ("gu",),
    "Gurmukhi": ("pa",),
    "Kannada": ("kn",),
    "Malayalam": ("ml",),
    "Han": ("zh", "ja"),
    "Hiragana": ("ja",),
    "Katakana": ("ja",),
    "Hangul": ("ko",),
    "Thai": ("th",),
}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "pl": "Polish",
    "sv": "Swedish", "tr": "Turkish", "vi": "Vietnamese", "id": "Indonesian",
    "ru": "Russian", "uk": "Ukrainian", "el": "Greek", "ar": "Arabic",
    "he": "Hebrew", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali",
    "ta": "Tamil", "te": "Telugu", "gu": "Gujarati", "pa": "Punjabi",
    "kn": "Kannada", "ml": "Malayalam", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "th": "Thai", "und": "unknown",
}

#: Every language code :func:`detect` can return, apart from ``"und"``.
SUPPORTED: Tuple[str, ...] = tuple(sorted(LANGUAGE_SCRIPT))

# Accented Latin letters each orthography actually uses.  Anything outside the
# set counts against the language.
ALLOWED_MARKS: Dict[str, FrozenSet[str]] = {
    "en": frozenset("é"),
    "es": frozenset("áéíóúüñ"),
    "fr": frozenset("àâäæçéèêëîïôöœùûüÿ"),
    "de": frozenset("äöüß"),
    "it": frozenset("àáèéìíîòóùú"),
    "pt": frozenset("áàâãçéêíóôõúü"),
    "nl": frozenset("áéëïöüèêç"),
    "pl": frozenset("ąćęłńóśźż"),
    "sv": frozenset("åäöé"),
    "tr": frozenset("âçğıöşüû"),
    "vi": frozenset(
        "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữự"
        "ỳýỷỹỵđ"
    ),
    "id": frozenset(),
}

# Characters that all but give the language away on sight.  Mostly letters,
# but not only: Spanish's inverted question and exclamation marks are the
# strongest visual cue the language has, and they are category Po.  The scorer
# counts the letters from the script-carrying characters of the text and the
# rest from the raw text, so both kinds actually do work.
DISTINCTIVE: Dict[str, FrozenSet[str]] = {
    "en": frozenset(),
    "es": frozenset("ñ¿¡"),
    "fr": frozenset("œ"),
    "de": frozenset("ß"),
    "it": frozenset(),
    "pt": frozenset("ãõ"),
    "nl": frozenset(),
    "pl": frozenset("ąćęłńśźż"),
    "sv": frozenset("å"),
    "tr": frozenset("ığş"),
    "vi": frozenset("ơưăđ"),
    "id": frozenset(),
    # Cyrillic: the four letters Ukrainian has and Russian does not, and the
    # four Russian has and Ukrainian does not.
    "uk": frozenset("іїєґ"),
    "ru": frozenset("ыэъё"),
    # Marathi uses this letter; Hindi essentially does not.
    "mr": frozenset("ळ"),
    "hi": frozenset(),
}

# Letters that rule a language out even though they share its script.
EXCLUDED: Dict[str, FrozenSet[str]] = {
    "ru": frozenset("іїєґ"),
    "uk": frozenset("ыэъё"),
}


def _build_word_weights() -> Dict[str, Dict[str, float]]:
    """A word shared by N languages is worth 1/N to each of them."""
    owners: Dict[str, int] = {}
    for words in STOPWORDS.values():
        for word in words:
            owners[word] = owners.get(word, 0) + 1
    return {
        lang: {word: 1.0 / owners[word] for word in words}
        for lang, words in STOPWORDS.items()
    }


def _build_marker_weights() -> Dict[str, Dict[str, float]]:
    owners: Dict[str, int] = {}
    for markers in MARKERS.values():
        for marker in markers:
            owners[marker] = owners.get(marker, 0) + 1
    return {
        lang: {marker: 1.0 / owners[marker] for marker in markers}
        for lang, markers in MARKERS.items()
    }


WORD_WEIGHTS: Dict[str, Dict[str, float]] = _build_word_weights()
MARKER_WEIGHTS: Dict[str, Dict[str, float]] = _build_marker_weights()
