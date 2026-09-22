"""Column-name hints: what a header suggests before a single value is read."""
from __future__ import annotations

import re
from typing import Dict, FrozenSet, Set

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT_RE = re.compile(r"[^a-z0-9]+")

SENSITIVE_TYPES = (
    "gender",
    "religion",
    "caste",
    "ethnicity",
    "nationality",
    "sexual_orientation",
    "disability",
    "political_opinion",
    "health_condition",
)

# (type, substrings of the squashed lowercase name, whole tokens)
_POSITIVE = (
    ("email", ("email", "mail"), ("email", "emailid", "mail")),
    (
        "phone",
        ("phone", "mobile", "telephone", "contact", "whatsapp", "msisdn", "cellular", "cellphone"),
        ("tel", "cell", "fax", "mob", "ph", "phno", "mobileno", "phoneno"),
    ),
    ("aadhaar", ("aadhaar", "aadhar", "adhaar", "adhar", "uidai"), ("uid",)),
    ("pan", ("pancard", "pannumber", "panno", "pannum"), ("pan",)),
    (
        "credit_card",
        ("creditcard", "debitcard", "cardnumber", "cardno", "cardnum", "ccnum", "ccnumber", "paymentcard"),
        ("cc", "card"),
    ),
    (
        "ipv4",
        ("ipaddress", "ipaddr", "clientip", "remoteip", "sourceip", "destip", "hostip", "serverip", "userip", "ipv4"),
        ("ip",),
    ),
    (
        "ipv6",
        ("ipaddress", "ipaddr", "clientip", "remoteip", "sourceip", "destip", "hostip", "serverip", "userip", "ipv6"),
        ("ip",),
    ),
    ("url", ("url", "website", "homepage", "href", "weblink", "hyperlink"), ("url", "link", "uri", "site", "web")),
    ("date_of_birth", ("dateofbirth", "birthdate", "birthday", "birth"), ("dob", "bday", "born", "birthday")),
    ("postal_code", ("pincode", "zipcode", "postalcode", "postcode", "postal"), ("pin", "zip", "pincode", "zipcode", "postcode")),
    ("address", ("address", "street", "locality", "residence"), ("addr", "address", "street")),
    (
        "person_name",
        (
            "firstname", "lastname", "fullname", "surname", "givenname", "familyname", "middlename",
            "customername", "contactname", "patientname", "employeename", "studentname", "personname",
            "holdername", "applicantname", "candidatename", "ownername", "fathername", "mothername",
            "spousename", "guardianname", "nomineename", "username", "accountname", "clientname",
            "authorname", "membername", "drivername", "passengername", "buyername", "sellername",
        ),
        ("name", "fname", "lname", "firstname", "lastname", "fullname", "surname", "forename"),
    ),
    ("gender", ("gender",), ("sex",)),
    ("religion", ("religion", "religious", "faith"), ()),
    ("caste", ("caste", "subcaste"), ()),
    ("ethnicity", ("ethnic",), ("race", "ethnicity")),
    ("nationality", ("nationality", "citizenship"), ()),
    ("sexual_orientation", ("sexualorientation", "sexuality"), ()),
    ("disability", ("disabil", "handicap", "differentlyabled"), ()),
    ("political_opinion", ("political", "politics", "partyaffiliation"), ()),
    ("health_condition", ("diagnosis", "disease", "medicalcondition", "healthcondition", "illness"), ()),
)

_PERSON_NAME_SUBS = next(subs for name, subs, _whole in _POSITIVE if name == "person_name")

# tokens that say "this `name` column is not a person"
_NOT_A_PERSON = frozenset(
    "product item sku file filename path table column col model brand city town country state province "
    "host hostname domain company org organisation organization team project dataset feature class label "
    "category cat plan store school university college bank branch package module function method variable "
    "field app application service server device machine job role title tag group event course subject drug "
    "medicine species breed color colour game movie film song album book vendor supplier merchant shop "
    "restaurant hotel airline airport station street road area region district zone plant animal dish "
    "recipe ingredient material tool part component version display screen bucket queue topic channel "
    "database db schema index key type kind format language lang currency unit metric field".split()
)

# tokens that say "amount, id or measurement" - a numeric column with one of these is not a phone
_ID_OR_AMOUNT = frozenset(
    "id ids uuid guid key pk fk idx index seq sequence no num number nbr amount amt price total subtotal "
    "sum qty quantity count cnt balance salary revenue cost fee fees rate score timestamp ts epoch time "
    "year age weight height lat lon lng latitude longitude version ver size length width duration value "
    "val code ref reference invoice order txn transaction serial batch lot page pages line rank level "
    "grade percent pct ratio avg mean median std min max delta diff distance km miles temp temperature "
    "pressure volume units unit".split()
)
_ID_WORD_RE = re.compile(r"[a-z]{2,}id")
_NOT_ID_WORDS = frozenset(
    "paid valid grid acid rapid solid fluid liquid bid kid lid mid rid skid void avoid raid braid maid "
    "said laid squid stupid vivid timid humid lucid rigid hybrid orchid pyramid unpaid prepaid postpaid "
    "invalid android druid fluid lipid plaid staid".split()
)


# tokens that mean "money or a measured quantity" - never a phone number, whatever the digits look like
_AMOUNT_WORDS = frozenset(
    "amount amt price total subtotal sum cost fee fees salary wage wages revenue balance payment "
    "paid due charge charges discount tax gst vat premium deposit withdrawal credit debit limit "
    "inr usd eur gbp rupees dollars cents paise value worth turnover profit loss income expense "
    "budget spend spends billing invoiceamount".split()
)


def amount_like(name: str) -> bool:
    """True when the header says the column holds money or a measured quantity."""
    return bool(tokens_of(name) & _AMOUNT_WORDS)


def tokens_of(name: str) -> FrozenSet[str]:
    """Lower-case tokens of a header: snake_case, camelCase, spaces and punctuation all split."""
    text = _CAMEL_RE.sub("_", str(name))
    toks = {t for t in _SPLIT_RE.split(text.lower()) if t}
    return frozenset(toks)


def squash(name: str) -> str:
    """Lower-case header with everything but letters and digits removed."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def hinted_types(name: str) -> Set[str]:
    """Every PII type the column name points at (may be empty)."""
    squashed = squash(name)
    toks = tokens_of(name)
    found: Set[str] = set()
    for pii_type, subs, whole in _POSITIVE:
        if any(s in squashed for s in subs) or any(t in toks for t in whole):
            found.add(pii_type)
    if "person_name" in found:
        strong = any(s in squashed for s in _PERSON_NAME_SUBS) or bool(
            toks & {"fname", "lname", "firstname", "lastname", "fullname", "surname", "forename"}
        )
        if not strong and toks & _NOT_A_PERSON:
            found.discard("person_name")
    if "address" in found:
        if found & {"ipv4", "email"} or "mac" in toks or "macaddress" in squashed:
            found.discard("address")
    if "sexual_orientation" not in found and "orientation" in toks and "sexual" in toks:
        found.add("sexual_orientation")
    if "email" in found and "mailing" in toks:  # mailing_address is an address, not an email
        found.discard("email")
    return found


def id_or_amount_like(name: str) -> bool:
    """True when the header says the column holds ids, amounts or measurements."""
    toks = tokens_of(name)
    if toks & _ID_OR_AMOUNT:
        return True
    return any(_ID_WORD_RE.fullmatch(t) and t not in _NOT_ID_WORDS for t in toks)


def hint_table(columns) -> Dict[str, Set[str]]:
    """Hints for every column of a frame, keyed by column name."""
    return {col: hinted_types(col) for col in columns}
