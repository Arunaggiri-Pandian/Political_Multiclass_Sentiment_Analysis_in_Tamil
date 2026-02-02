# Shared Task on Political Multiclass Sentiment Analysis of Tamil X(Twitter) Comments

**DravidianLangTech @ ACL 2026**

Sentiment analysis is a vital task in Natural Language Processing (NLP) that seeks to identify and categorize opinions expressed in text into predefined classes. In the political context, understanding public sentiment is crucial for getting opinions, addressing concerns, and shaping policies. The increasing prominence of social media platforms like X (Twitter) has made them rich repositories of real-time, diverse and expressive political discourse. This shared task focuses on Political Multiclass Sentiment Analysis of Tamil Comments (tweets) collected from Twitter (X).

## Use Cases

The goal of this task is to classify political sentiments expressed in Tamil tweets into the following seven distinct categories:

1. **Substantiated (ஆதாரப்பூர்வமானது)**
2. **Sarcastic (கிண்டல்)**
3. **Opinionated (தனிப்பட்டக்கருத்து)**
4. **Positive (நேர்மறை)**
5. **Negative (எதிர்மறை)**
6. **Neutral (நடுநிலை)**
7. **None of the above (எதுவும்இல்லை)**

## Explanation of Categories

### Substantiated (ஆதாரப்பூர்வமானது)

This category contains tweets that provide information supported by evidence such as links, references, or factual claims. The quality of evidence is not important; only the presence of support matters.

**Example:**
> "தேர்தல் வாக்குறுதி எண்.261-ல் வறுமைக்கோட்டிற்குக் கீழே உள்ள குடும்பங்களைச் சேர்ந்த 1 லட்சம் கிராமப்புறப் பெண்களுக்குக் கால்நடைகள் வளர்ப்பு, மீன் பிடித்தல், வண்ண மீன் வளர்ப்பு, மண் பானைகள் செய்தல் போன்ற விவசாயம் சார்ந்த சிறிய தொழில்கள் மற்றும் வணிகம் செய்வதற்கு வட்டியில்லாக் கடனாக ரூ.50,000 வழங்கப்படும் என்று கூறிவிட்டு இதுவரை வழங்காதது ஏன்..? பதிலுக்காகக் காத்திருப்போம்..! கேள்விகள் தொடரும்..!"

**English Translation:**
> "The election promise No. 261 stated that interest-free loans of ₹50,000 would be provided to one lakh rural women from families below the poverty line for small agriculture-related occupations and businesses such as livestock rearing, fishing, ornamental fish cultivation, and pottery making. Why has this not been provided so far? We will wait for an answer. The questions will continue."

### Sarcastic (கிண்டல்)

Tweets that use sarcasm, mockery, or irony, often where positive words convey negative criticism in a humorous or ridiculing way.

**Example:**
> "சிவப்பா இருந்திருந்தா இந்தியாவுக்கு நான்தான் பிரதமர்'?சீமான் கலகல பேச்சு!"

**English Translation:**
> "If only I had been fair-skinned, I would have been the Prime Minister of India." Seeman's humorous remark.

### Opinionated (கருத்துச்சாரம் / கொள்கைப் பிடிவாதம் கொண்ட)

Tweets expressing strong personal views, subjective interpretations, or personally biased beliefs about political figures or situations.

**Example:**
> "திமுக ஆட்சியில் சாலைப் பராமரிப்பு மிகவும் மோசமாக உள்ளது."

**English Translation:**
> "Under DMK rule, road maintenance is very poor."

### Positive (நேர்மறை)

Tweets showing approval, optimism, or appreciation toward a political leader, party, event, or decision—without personal bias.

**Example:**
> "கோவை நாடாளுமன்றத் தொகுதி வெற்றி வேட்பாளர் அண்ணனின் தங்கை, எங்களின் சகோதரி கலாமணி அவர்களுக்கு மைக் சின்னத்தில் வாக்களிப்போம்!.."

**English Translation:**
> "We will vote for the microphone symbol for Kalamani, the sister of our elected brother from the Coimbatore parliamentary constituency."

### Negative (எதிர்மறை)

Tweets expressing criticism, dissatisfaction, or highlighting failures of political leaders or actions — typically based on commonly known facts, not personal bias.

**Example:**
> "தமிழ்நாட்டிற்கு தேவையில்லாத குப்பை இந்த காங்கிரஸ்,பாஜக"

**English Translation:**
> "Congress and BJP are useless garbage that Tamil Nadu does not need."

### Neutral (நடுநிலை)

Tweets that provide general political information or factual updates without expressing opinions, emotions, or bias.

**Example:**
> "மதுரை தொகுதியில் அதிமுக வேட்பாளர் சரவணனை ஆதரித்து எம்.எல்.ஏ ராஜன் செல்லப்பா பிரச்சாரம் செய்கிறார்"

**English Translation:**
> "MLA Rajan Chellappa campaigns in support of AIADMK candidate Saravanan in Madurai constituency."

### None (எதுவும் இல்லை)

Tweets unrelated to politics and not fitting into any of the defined categories. These are off-topic statements.

**Example:**
> "இன்று வானிலை மிகவும் நன்றாக உள்ளது"

**English Translation:**
> "The weather is very nice today."

## Dataset

We providing training, development, and test datasets to the participants in Tamil language. Given a Twitter (X) comment, systems have to classify it into one of the sentiment categories that mentioned above.

The participants will be provided development, training and test dataset. To download the data and participate, click "Participate" tab. As far as we know, this is the first shared task on multiclass sentimental analysis of political tweets in Tamil.
