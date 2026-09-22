"""Word lists used by the clarity, structure and repetition measures.

Every list here is a documented heuristic, not a linguistic model. They are
deliberately small and English-centric; see the README for what that means for
text in other languages.
"""
from __future__ import annotations

from typing import FrozenSet

STOPWORDS: FrozenSet[str] = frozenset("""
a about above after again against all almost also am an and another any are as at
be because been before being below between both but by
can cannot could did do does doing down during
each either else enough even ever every
few for from further
had has have having he her here hers herself him himself his how however
i if in into is it its itself
just
me might more most much must my myself
near no nor not now
of off on once only or other others ought our ours ourselves out over own
per
rather really
same shall she should so some still such
than that the their theirs them themselves then there these they this those though through to too
under until up upon us
very
was we were what when where whether which while who whom whose why will with within without would
yet you your yours yourself yourselves
""".split())

TRANSITION_WORDS: FrozenSet[str] = frozenset("""
accordingly additionally afterwards also alternatively although besides but
consequently conversely
finally first firstly further furthermore
hence however
indeed instead
later likewise
meanwhile moreover
nevertheless next nonetheless
overall
second secondly similarly since so specifically still subsequently
then therefore thereafter third thirdly though thus
ultimately unlike
whereas while yet
""".split())

TRANSITION_PHRASES = (
    "as a result",
    "at the same time",
    "by contrast",
    "even so",
    "for example",
    "for instance",
    "in addition",
    "in conclusion",
    "in contrast",
    "in fact",
    "in other words",
    "in particular",
    "in short",
    "in summary",
    "on the other hand",
    "on the whole",
    "that said",
    "to sum up",
    "to summarise",
    "to summarize",
)

FILLER_WORDS: FrozenSet[str] = frozenset("""
absolutely actually basically certainly clearly completely definitely
entirely essentially extremely fairly highly incredibly
just literally obviously particularly perfectly pretty
quite really remarkably simply somewhat
totally truly utterly very virtually
""".split())

FILLER_PHRASES = (
    "at this point in time",
    "due to the fact that",
    "in order to",
    "it is important to note that",
    "it should be noted that",
    "needless to say",
    "the fact that",
    "there is no doubt that",
)

HEDGE_WORDS: FrozenSet[str] = frozenset("""
apparently approximately arguably could generally largely may maybe might
mostly often partly perhaps possibly potentially presumably probably
relatively seemingly seems sometimes somewhat suggests tends typically usually
""".split())

BE_FORMS: FrozenSet[str] = frozenset(
    "is are was were be been being am get gets got getting gotten".split()
)

IRREGULAR_PARTICIPLES: FrozenSet[str] = frozenset("""
been begun bought broken brought built caught chosen come cut done drawn driven
drunk eaten fallen felt fed found forgotten found given gone gotten grown had
heard held hidden hit hurt kept known laid led left lent let lost made meant met
paid put read ridden risen run said seen sent set shown shut sold sent spent
spoken spread stolen struck sung sunk sworn taken taught thought thrown told
understood woken worn won written
""".split())

# A nominalisation is a noun built OUT OF A VERB - "implementation" from "implement",
# "assessment" from "assess". Telling a writer to unpack one is good advice.
#
# These suffixes are reliably verb-derived, so a bare suffix match is safe enough:
NOMINALISATION_SUFFIXES = ("ization", "isation", "tion", "sion", "ment")

# -ance and -ence are NOT. "distance", "patience", "science", "silence", "sentence",
# "audience" and "experience" all end that way and none is built from a verb, so a bare
# suffix match fires on ordinary prose. Only these, where an English verb really is the
# stem, count:
VERB_DERIVED_ANCE_ENCE = frozenset({
    "acceptance", "acquaintance", "adherence", "allowance", "appearance", "assistance",
    "attendance", "avoidance", "clearance", "coherence", "compliance", "conformance",
    "continuance", "correspondence", "dependence", "disturbance", "emergence",
    "endurance", "existence", "governance", "guidance", "inference", "insurance",
    "interference", "issuance", "maintenance", "observance", "occurrence", "performance",
    "persistence", "preference", "reference", "reliance", "remembrance", "resistance",
    "tolerance",
})

# -ity, -ness and -ism are dropped altogether. They build nouns out of ADJECTIVES and
# nouns, not verbs: "kindness" from "kind", "quality" from "qualis", "tourism" from
# "tour". Flagging them told writers that clean prose was bureaucratic, and a check that
# fires on good writing is worse than no check, so this measure now prefers to miss a
# real nominalisation over inventing one.

# Common concrete nouns that end in a nominalisation suffix but are not the
# style problem the measure is looking for. "The configuration file lives in
# your home directory" is not a verb hiding inside a noun, and telling a writer
# to turn "documentation" back into a verb is bad advice, so ordinary everyday
# nouns are listed out here.
NOMINALISATION_EXCEPTIONS: FrozenSet[str] = frozenset("""
application association attention attraction
business
caption cement collection comment community competition condition configuration
conference connection constitution convention conversation corporation
department destination dimension direction document documentation
edition education election element emotion environment equipment exception
exhibition experiment expression extension
fashion fiction foundation fraction function
generation
illustration information installation institution instruction instrument
introduction invention
junction
location
mansion million mission moment motion
nation notion
occasion operation option organisation organization
partition passion pension percussion permission population portion position
precision profession proportion
question
reaction reception region relation religion reputation
section session situation station
tension tradition transition
vacation version vision
witness
""".split())

#: Past participles that normally work as adjectives after "to be": "he was
#: tired", "she is qualified for the role". Without an agent these describe a
#: state rather than an action done to the subject, so the clarity measure
#: leaves them alone. An explicit agent always wins, so "the door was locked by
#: the guard" is still counted as passive.
PREDICATE_ADJECTIVES: FrozenSet[str] = frozenset("""
accustomed amazed amused annoyed ashamed astonished attached
based bored
closed committed complicated concerned confused convinced crowded
dedicated delighted depressed determined devoted disappointed dissatisfied
divorced done dressed drunk
educated embarrassed engaged excited exhausted experienced
fascinated finished frightened frustrated
gifted gone
inclined interested involved irritated
limited located locked
married mistaken mixed motivated
opposed
packed pleased prepared puzzled
qualified
related relaxed relieved retired
satisfied scared shocked situated skilled stressed suited supposed surprised
talented terrified thrilled tired troubled
unemployed
worn worried
""".split())

#: A degree adverb straight before the word settles it: "was very tired" is an
#: adjective, because no real passive takes "very" ("was very written" is not
#: English). This is the standard test for an adjectival past participle.
DEGREE_ADVERBS: FrozenSet[str] = frozenset("""
awfully deeply enormously extremely fairly
highly hugely incredibly intensely
least less
most much
particularly pretty
quite rather really remarkably
so somewhat
terribly too totally truly
utterly
very
""".split())

#: Words that may sit between a determiner and the noun of a "by" phrase.
AGENT_DETERMINERS: FrozenSet[str] = frozenset("""
a an the this that these those
another any both each either every many most other several some such
her his its my our their your
""".split())

#: "by" followed by one of these is an adverbial phrase, not the doer of an
#: action: "the room was packed by nine" is a deadline, not an agent.
NON_AGENT_AFTER_BY: FrozenSet[str] = frozenset("""
accident car chance choice default definition design email fax hand hands
letter mail mistake phone plane post road sea ship train
one two three four five six seven eight nine ten eleven twelve
half quarter dozen
now then today tomorrow yesterday tonight midnight midday noon
monday tuesday wednesday thursday friday saturday sunday
january february march april may june july august september october november december
morning afternoon evening night day week weekend month year
autumn fall spring summer winter
christmas easter
""".split())

ABBREVIATIONS: FrozenSet[str] = frozenset("""
al approx apr assn aug ave blvd bros capt cf co col comp corp dept dr eds eg
esp est etc feb fig figs gen gov hon ibid inc jan jr jul jun ltd mar mr mrs ms
mt mts no nos nov oct pl pp prof rd rep rev sen sep sept sgt sr st univ vs vol
""".split())
