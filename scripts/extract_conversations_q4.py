import os
import pandas as pd
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../P1_DATA")

def read_dev():
    path = os.path.join(DATA_DIR, "trac2_CONVT_dev.csv")
    return pd.read_csv(
        path,
        engine="python",
        sep=",",
        quotechar='"',
        escapechar='\\',
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="skip"  # Skip malformed lines
    )

if __name__ == "__main__":
    dev_df = read_dev()
    
    # Find conversations with 10+ turns
    conv_counts = dev_df.groupby('conversation_id').size()
    valid_convs = conv_counts[conv_counts >= 10].index[:5]
    
    print("="*60)
    print("Q4: EXTRACTING 5 CONVERSATIONS FOR LLM PROMPTING")
    print("="*60)
    print(f"\nSelected Conversation IDs: {list(valid_convs)}\n")
    
    for i, conv_id in enumerate(valid_convs, 1):
        conv = dev_df[dev_df['conversation_id'] == conv_id].head(10)
        print(f"\n{'='*60}")
        print(f"CONVERSATION {i} (ID: {conv_id})")
        print('='*60)
        
        for idx, row in conv.iterrows():
            print(f"\nTurn {row['turn_id']}: {row['text']}")
            print(f"  [Ground Truth: Emotion={row['Emotion']}, Polarity={row['EmotionalPolarity']}, Empathy={row['Empathy']}]")
    
    print("\n" + "="*60)
    print("Copy the conversations above for your LLM prompts!")
    print("="*60)
