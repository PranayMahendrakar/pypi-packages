"""A small English lexicon used to canonicalize wording before comparing.

This is deliberately tiny and hand written: it maps a few dozen everyday
paraphrase families onto one token each, so that "postponed", "pushed back"
and "moved to a later date" land on the same word before any vectorizing
happens. It is an approximation of meaning, not a model of it - see the
README, and pass ``embed=`` when real semantics matter.

Negations ("not", "never", "no", ...) are deliberately NOT stopwords: dropping
them would make a sentence and its opposite look identical.
"""
from __future__ import annotations

from typing import Dict, Set

#: Words carrying no topical content. Negations are excluded on purpose.
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
    "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "may", "me", "might", "more", "most", "much", "must", "my", "myself",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "per", "same", "shall", "she", "should",
    "so", "some", "still", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "therefore", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "upon", "us", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "whose", "why", "will", "with", "would", "you", "your", "yours", "yourself",
    "yourselves",
}

#: Multi-word phrases mapped onto one canonical token. Longest match wins.
PHRASES: Dict[str, str] = {
    "at a later date": "postpone",
    "to a later date": "postpone",
    "later date": "postpone",
    "another time": "postpone",
    "pushed back": "postpone",
    "push back": "postpone",
    "moved back": "postpone",
    "put off": "postpone",
    "called off": "cancel",
    "call off": "cancel",
    "kicked off": "start",
    "kick off": "start",
    "wrapped up": "finish",
    "wrap up": "finish",
    "went up": "increase",
    "go up": "increase",
    "gone up": "increase",
    "went down": "decrease",
    "go down": "decrease",
    "gone down": "decrease",
    "a lot of": "many",
    "lots of": "many",
    "a number of": "many",
    "as well as": "and",
    "due to": "because",
    "owing to": "because",
    "in order to": "to",
    "figure out": "understand",
    "find out": "learn",
    "look into": "investigate",
    "looked into": "investigate",
    "look for": "search",
    "set up": "setup",
    "sign up": "register",
    "signed up": "register",
    "log in": "login",
    "sign in": "login",
    "turn off": "disable",
    "turned off": "disable",
    "turn on": "enable",
    "turned on": "enable",
    "shut down": "stop",
    "broke down": "fail",
    "break down": "fail",
    "gave up": "quit",
    "give up": "quit",
    "took place": "happen",
    "take place": "happen",
    "carried out": "perform",
    "carry out": "perform",
    "made up of": "contain",
    "consists of": "contain",
    "consist of": "contain",
    "right away": "immediately",
    "at once": "immediately",
    "in addition": "also",
    "for example": "example",
    "for instance": "example",
    "no longer": "not",
    "used to": "formerly",
}


#: When a phrase already names the event, the generic verb that introduces it
#: adds nothing: "moved the meeting to a later date" *is* "postponed the
#: meeting". Keyed by the canonical token, valued by the verbs it absorbs.
ABSORBED_VERBS: Dict[str, Set[str]] = {
    "postpone": {
        "move", "moved", "moves", "moving",
        "push", "pushed", "pushing",
        "shift", "shifted", "shifting",
        "bump", "bumped", "bumping",
        "set", "sets", "setting",
    },
}


def _spread(groups: Dict[str, str]) -> Dict[str, str]:
    """Expand ``{canonical: "word word word"}`` into ``{word: canonical}``."""
    out: Dict[str, str] = {}
    for canonical, words in groups.items():
        for word in words.split():
            out[word] = canonical
    return out


#: Single words mapped onto one canonical token per meaning family.
SYNONYMS: Dict[str, str] = _spread({
    "postpone": "postponed postpone postpones postponing delayed delay delays "
                "deferred defer rescheduled reschedule rescheduling",
    "cancel": "cancelled canceled cancel cancels cancelling canceling scrapped scrap aborted abort",
    "start": "started start starts starting began begin begins begun commenced "
             "commence initiated initiate launched launch launches",
    "finish": "finished finish finishes completed complete completes completing "
              "concluded conclude ended end ends done",
    "increase": "increased increase increases increasing rose rise rises risen grew "
                "grow grows grown climbed climb higher",
    "decrease": "decreased decrease decreases decreasing fell fall falls fallen dropped "
                "drop drops declined decline declines lower reduced reduce reduces shrank shrink",
    "buy": "bought buy buys buying purchased purchase purchases purchasing",
    "sell": "sold sell sells selling sale sales",
    "help": "helped help helps helping assist assisted assists aid aided support supported supports",
    "fix": "fixed fix fixes fixing repaired repair repairs resolved resolve resolves "
           "corrected correct corrects patched patch",
    "problem": "problem problems issue issues bug bugs defect defects fault faults "
               "error errors glitch glitches",
    "company": "company companies firm firms business businesses organisation organisations "
               "organization organizations corporation corporations",
    "employee": "employee employees staff worker workers personnel",
    "customer": "customer customers client clients buyer buyers",
    "money": "money cash funds funding",
    "price": "price prices cost costs pricing fee fees charge charges",
    "car": "car cars automobile automobiles vehicle vehicles",
    "doctor": "doctor doctors physician physicians",
    "movie": "movie movies film films",
    "job": "job jobs position positions role roles vacancy vacancies",
    "home": "home homes house houses residence",
    "say": "said say says saying stated state states mentioned mention mentions "
           "noted note notes remarked remark told tell tells",
    "think": "think thinks thought believe believed believes feel felt reckon",
    "want": "want wants wanted wish wished wishes desire desired",
    "need": "need needs needed require required requires requirement requirements",
    "show": "show shows showed shown display displays displayed demonstrate demonstrated illustrate",
    "make": "make makes made create created creates build built builds produce produced generate generated",
    "use": "use uses used using utilize utilized utilise employ employed",
    "get": "get gets got obtain obtained receive received gain gained",
    "big": "big large huge massive enormous",
    "small": "small little tiny minor",
    "good": "good excellent nice fine positive",
    "bad": "bad poor terrible awful horrible negative",
    "fast": "fast quick quickly rapid rapidly swift swiftly speedy",
    "slow": "slow slowly sluggish",
    "often": "often frequently frequent regularly usually",
    "maybe": "maybe perhaps possibly probably likely",
    "important": "important critical crucial essential vital significant",
    "difficult": "difficult hard tough challenging tricky",
    "easy": "easy simple straightforward easily",
    "happy": "happy glad pleased delighted satisfied",
    "sad": "sad unhappy upset disappointed",
    "angry": "angry mad furious annoyed frustrated",
    "child": "child children kid kids",
    "learn": "learn learns learned learnt study studied studies",
    "teach": "teach teaches taught instruct train trained",
    "answer": "answer answers answered reply replied replies respond responded response responses",
    "ask": "ask asks asked question questions questioned query queries inquire",
    "find": "find finds found discover discovered locate located",
    "change": "change changes changed modify modified alter altered adjust adjusted update updated",
    "remove": "remove removes removed delete deleted deletes erase erased",
    "add": "add adds added insert inserted append appended",
    "check": "check checks checked verify verified validate validated test tested",
    "send": "send sends sent deliver delivered ship shipped dispatch dispatched",
    "meet": "meeting meetings meet meets met",
    "document": "document documents doc docs report reports paper papers",
    "data": "data information info details",
    "result": "result results outcome outcomes finding findings",
    "plan": "plan plans planned planning schedule scheduled schedules",
    "project": "project projects initiative initiatives",
    "team": "team teams crew",
    "goal": "goal goals objective objectives target targets aim aims",
    "risk": "risk risks danger dangers threat threats hazard hazards",
    "benefit": "benefit benefits advantage advantages upside",
    "method": "method methods approach approaches technique techniques way ways",
})
