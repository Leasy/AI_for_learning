"""Frozen original bilingual message-label rules. No treatment or outcome input."""
import re
import unicodedata

def norm(s):
    s=re.sub(r'\\?u([0-9a-fA-F]{4})',lambda m:chr(int(m[1],16)),str(s))
    s=s.encode('utf-16','surrogatepass').decode('utf-16','replace')
    s=s.replace('ı','i').replace('İ','I').replace('’',"'").replace('\\', '')
    s=''.join(c for c in unicodedata.normalize('NFKD',s.lower()) if not unicodedata.combining(c))
    s=re.sub(r'\s+',' ',s).strip(' \"\'')
    return s

def tokens(s): return re.findall(r'[a-z]+|\d+',s)
def is_copy(s,target):
    a=tokens(s); b=set(tokens(target))
    return len(a)>=10 and len(set(a)&b)/max(1,len(set(a)))>=.88

TEMPLATES=[
    'can you help me figure out how to solve this problem',
    'can you help me better understand the question',
    'can you help me understand the question',
]
CONCEPT=r'(?:even|odd|prime|natural|real|integer|rational|irrational|complex|positive|negative|number|function|domain|range|slope|gradient|parallel|perpendicular|intercept|coefficient|constant|linear|quadratic|polynomial|proposition|negation|conjunction|disjunction|implication|conditional|contrapositive|converse|truth|tautology|subset|set|combination|permutation|factorial|probability|intersection|union|complement|mutually|independent|vertex|discriminant|equation|fraction|exponent|power|root|logarithm|ceiling|floor|modulo|absolute|bijective|injective|surjective|asal|cift|tek|fonksiyon|tanim|deger|egim|kume|onerme|olasilik|kombinasyon|permutasyon|faktoriyel|denklem|katsayi|kesisim|birlesim|mantik)'
MATH_ACTION=r'(?:add|subtract|multiply|multipli|divide|divid|factor|substitut|simplif|cancel|expand|equal|negative|positive|choose|count|square|cube|power|exponent|slope|domain|range|function|equation|denominator|numerator|payda|carp|bol|topla|cikar|esit|denklem|kume|fonksiyon)'

def classify(s,system='',previous_assistant=''):
    if not s: return 'ambiguous','empty'
    if any(s.rstrip('?.!')==t for t in TEMPLATES) or 'fill in' in s:
        return 'ambiguous','template'
    if re.search(r'you are (?:a |an )?(?:math|helpful|chatgpt)|i(?: am|.m) a (?:math|maths) teacher|ignore (?:all|previous)|role.?play|act as|forget (?:all|your)',s):
        return 'ambiguous','role_instruction'
    # Strong direct-answer requests survive pasted problem text.
    only=re.search(r'(?:only|just)\s+(?:give|tell|write|say|show|the|answer)|(?:no|without|dont|don.t)\s+(?:explain|explanation|steps)|sadece\s+(?:cevap|cevab|sonuc)|(?:cevabi|sonucu)\s+(?:soyle|ver|yaz)',s)
    if only and re.search(r'answer|result|solution|cevap|cevab|sonuc',s):
        if not re.search(r'don.t (?:just |only )?(?:give|tell)|not (?:just|only)',s): return 'A','explicit_answer_only'
    if is_copy(s,system): return 'ambiguous','problem_copy'
    if is_copy(s,previous_assistant): return 'ambiguous','assistant_overlap'
    # Copied/typed problem statements can be pasted in a different question's chat.
    if len(tokens(s))>=30 and re.search(r'given|let |find |how many|what is the number|mathematics lesson|teacher asked',s):
        return 'ambiguous','long_problem_statement'
    # Negated request for answers is not evidence of answer-seeking.
    if re.search(r'(?:don.t|do not|without) (?:give|tell|show).{0,25}answer|cevabi (?:verme|soyleme)',s):
        return 'U','requests_help_without_answer'
    if re.fullmatch(r'what is the (?:equation|function)[ ?!.]*',s): return 'ambiguous','unspecified_result'
    if re.search(r'^why are you.{0,35}(?:wrong|lying|stupid|dumb)',s): return 'ambiguous','unspecified_complaint'
    if re.search(r'^(?:how (?:do|can) i find the answer|(?:can you )?help me.{0,10}solve)',s):
        return 'ambiguous','generic_solution_help'
    if re.search(r'\b(?:why|neden|niye|nicin)\b',s) and len(tokens(s))>=3 and (re.search(CONCEPT+'|'+MATH_ACTION,s) or re.search(r'\d',s)):
        return 'U','specific_why_question'
    if re.search(r'\b(?:how (?:did|do|does|can|is|are|come)|where (?:did|does))\b',s) and (re.search(MATH_ACTION,s) or re.search(r'\d',s)):
        if not re.search(r'how (?:do|can) (?:i|you|we) (?:solve|find the answer)',s):
            return 'U','specific_how_step'
    if re.search(r'(?:what is|whats|what.s)\s+(?:the\s+)?(?:(?:real|correct|final)\s+)?(?:answer|answeer)\b|(?:what is|whats|what.s)\s+(?:the\s+)?(?:(?:numeral|maximum|minimum)\s+)?(?:value|number)\b|what (?:is|are).{0,20}(?:combination|prime factors|domain|range|truth value).{0,8}(?:of|\d)|degeri nedir|en buyuk domain',s):
        return 'A','specific_result_request'
    if re.search(r'\b(?:explain|meaning|mean by|understand)\b',s) and re.search(r'\b'+CONCEPT+r'\b',s):
        return 'U','concept_explanation'
    if re.search(r'\b(?:what does|what do)\b.{0,55}\bmean',s) or re.search(r'what is the (?:formula|rule|method|definition)',s):
        return 'U','concept_definition'
    if re.fullmatch(r'(?:so )?(?:what is|what are|whats|what.s) (?:a |an |the )?'+CONCEPT+r'(?: (?:number|numbers|function|functions|law|laws|rule|rules))?s?(?: in (?:turkish|english))?[ ?!.]*',s):
        return 'U','concept_definition'
    if re.search(r'\b'+CONCEPT+r'\b',s) and re.search(r'\b(?:ne demek|anlat|acikla|anlamadim)\b',s):
        return 'U','concept_explanation_tr'
    if re.fullmatch(CONCEPT+r'(?: sayi| sayilar)? (?:nedir|ne demek)[ ?!.]*',s): return 'U','concept_definition_tr'
    if re.search(r'(?:explain|don.t understand|dont understand|didn.t understand|didnt understand|anlamadim|acikla).{0,35}(?:step|part|line|operation|adim|islem)|(?:step|adim).{0,15}(?:mean|anlam|explain)',s):
        if not re.search(r'(?:step by step|adim adim)',s): return 'U','specific_step_explanation'
    if re.search(r'\b(?:because|therefore|since|hence|cunku|dolayisiyla)\b',s) and (re.search(MATH_ACTION,s) or re.search(r'\d',s)) and len(tokens(s))>=5:
        return 'U','articulated_reasoning'
    if re.search(r'\b(?:i|we) (?:have |had |already )?(?:tried|calculated|multiplied|divided|subtracted|added|substituted|factored|used)\b',s) and len(tokens(s))>=5 and (re.search(MATH_ACTION,s) or re.search(r'\d',s)):
        return 'U','described_attempt'
    if re.search(r'(?:my|the|your) (?:mistake|error)|where (?:did|am) i.{0,20}wrong|what (?:did|am) i.{0,20}wrong|hatam|hata nerede',s):
        return 'U','error_diagnosis'
    if re.search(r'\b(?:shouldn.t|wouldn.t|should we|do we need|dont we|don.t we)\b',s) and re.search(MATH_ACTION,s):
        return 'U','reasoning_check'
    if re.search(r'(?:step.by.step|adim adim|explain (?:the )?(?:solution|answer)|show (?:the |me )?(?:steps|solution))',s):
        return 'ambiguous','worked_solution_request'
    if re.search(r'(?:what|whats|what.s|give|tell|show|write|say|calculate|find).{0,30}(?:answ\w*|final result)|(?:answer|result)\s*(?:only|please|pls|plz)|(?:cevap|cevabi|cevabin|sonuc|sonucu)\s*(?:ne|nedir|ver|soyle|yaz)|(?:soyle|ver|yaz).{0,15}cevab',s):
        if not re.search(r'^(?:is|isn.t|could|can|i think|i found|the answer is|my answer)',s): return 'A','answer_request'
    if re.fullmatch(r'(?:the )?(?:answer|answ\w*|cevap|cevabi|sonuc)[ ?!.]*',s): return 'A','answer_request'
    if re.search(r'\b(?:solve|solv|sove|coz|cozer|cozebilir)\w*\b|(?:can|could|will) (?:you|u) do (?:it|this|the)',s) and len(tokens(s))<=20:
        if not re.search(r'\b(?:how|nasil|why|neden|learn|understand|ogren|anla)\b|(?:i|we) (?:cant|can.t|cannot|couldnt|couldn.t|can not)|did i |cozemiyorum|cozemedim|cozemem',s): return 'A','delegate_solution'
    if re.search(r'^(?:is (?:the |my )?answer|the answer is|i think|is it|are you sure|emin misin)',s) or re.fullmatch(r'[-+\d\s.,/=()xyzabc?]+',s):
        return 'ambiguous','answer_guess_or_confirmation'
    return 'ambiguous','other_or_generic'

