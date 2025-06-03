# DD2417_Pairing-news-items-and-headlines_Group-19

## Requirements
- Python 3.9 or later
- pip (Python package installer)

## Running the program:
1. Install required packages:
```bash
pip install -r requirements.txt
```
2. Download the precomputed word2vec model and place in the same folder, link and name of file is displayed in bash command:

```bash
curl -L -o GoogleNews-vectors-negative300.bin \https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin
```

3. Run the script:
Optional, specify the matching direction 'title_to_article' or 'article_to_title'. if not specified the script 
defaults to 'title_to_article'. 

```bash
python first.py title_to_article
python first.py article_to_title
```
 