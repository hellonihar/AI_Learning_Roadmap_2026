# Implementation — Online Course Recommender

## 1. Skill‑TF-IDF Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

courses['skills_str'] = courses['skills'].apply(' '.join)
vec = TfidfVectorizer(tokenizer=lambda x: x.split(), lowercase=False)
course_skill_matrix = vec.fit_transform(courses['skills_str'])

def skill_match(user_skills, course_idx):
    user_vec = vec.transform([' '.join(user_skills)])
    sim = cosine_similarity(user_vec, course_skill_matrix[course_idx]).flatten()
    return sim[0]
```

## 2. Two‑Tower Embedding Model

```python
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

user_emb = tf.keras.layers.Embedding(n_users, 64)
course_emb = tf.keras.layers.Embedding(n_courses, 64)
dot = tf.keras.layers.Dot(axes=1)([user_emb, course_emb])
model = tf.keras.Model(inputs=[user_input, course_input], outputs=dot)
model.compile(optimizer='adam', loss='binary_crossentropy')
```

## 3. Hybrid Predictor

```python
def recommend(user_id, user_skills=None, k=5):
    if user_skills or user_has_few_enrollments(user_id):
        # Cold-start: skill-match + popularity boost
        scores = 0.7 * skill_scores + 0.3 * popularity_scores
    else:
        # Warm: collaborative embedding scores
        scores = two_tower_model.predict([user_id, all_course_ids])
    return top_k_courses(scores, k)
```
