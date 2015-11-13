import re
from nltk.corpus import stopwords

NEG_INF = float('-inf')
STOP_WORDS = set(stopwords.words('english'))
PAREN_EXPRESSION = re.compile('\s*\([^)]*\)\s*')
