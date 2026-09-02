## Movie Recommendation System using NLP

A content based movie recommendation system built using Python, Natural Language Processing (NLP), and cosine similarity. The system analyzes movie metadata such as plot overview, genres, keywords, cast, crew to recommend movies that are most similar to a movie selected by the user.

# Project Overview
The project uses the TMDB 5000 Movies and TMDB 5000 Credits datasets. Movie information from the two datasets is merged and relevant textual features are processed to create a single tag representation for each movie. 
The movie dataset contains fields such as budget, genres, keywords, overview, popularity, release date, revenue, title, vote average, and vote count.

# The recommendation pipeline includes:
Loading and inspecting the movie and credits datasets.
Cleaning and merging the datasets.
Selecting relevant recommendation features.
Converting JSON-like genre and keyword fields into usable lists.
Combining movie overview, genres, and keywords into a single text field.
Applying stemming and converting text to lowercase.
Vectorizing movie tags using Bag of Words algorithm
Calculating movie-to-movie similarity using cosine similarity.

Returning the top 10 movies most similar to a selected movie.



The credits dataset contains movie ID, title, cast, and crew information.

After preprocessing and merging, the project works with approximately 4,804 movies for the recommendation pipeline. The original movie dataset contains 4,807 rows, while the merged dataset initially contains 4,545 rows before further processing. fileciteturn0file0L136-L164 fileciteturn0file0L183-L186

Features Used

The project initially identifies the following fields as relevant to building the recommendation system:
movie_id, title, overview, genres, keywords, cast, crew

Due to corrupt/inconsistent data in the credits file, the project removes cast and director information from the final recommendation features. The final feature set used is: movie_id, title, overview, genres, keywords


# Methodology
**1. Data Loading and Merging** 
The datasets are loaded using Pandas and merged using the movie title.
    movie_df = pd.read_csv('tmdb_5000_movies.csv')
    credit_df = pd.read_csv('tmdb_5000_credits.csv')
    df = movie_df.merge(credit_df, on="title")

**2. Data Cleaning**
Only the fields required for recommendation are retained. Missing movie overviews are removed because only a small number of records have missing values.
The project also checks for duplicate records and missing values during preprocessing. 

**3. Processing Genres and Keywords** Genres and keywords are stored as JSON-like strings. Python's ast.literal_eval() is used to convert these strings into Python lists, after which the relevant names are extracted.

**4. Creating the Movie Tags** 
The movie overview is tokenized, and the overview, genres, and keywords are combined into one tag field.
df["overview"] = df["overview"].apply(lambda x: x.split())
df["tag"] = df["overview"] + df["genres"] + df["keywords"]
new_df = df[["movie_id", "title", "tag"]]
new_df["tag"] = new_df["tag"].apply(lambda x: " ".join(x))

This creates a single text representation of each movie that can be compared with other movies. 

**5. NLP Preprocessing** Porter Stemming is applied to reduce words to their stems, followed by conversion of the text to lowercase.

**6. Text Vectorization** The processed tags are converted into numerical vectors using Scikit-learn's CountVectorizer.
The project limits the vocabulary to the 3,000 most relevant features and removes English stop words to reduce the sparsity of the feature matrix.

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(
    max_features=3000,
    stop_words="english"
)

tag_vectors = cv.fit_transform(new_df["tag"]).toarray()

The resulting vector matrix has the shape (4804, 3000). fileciteturn0file0L916-L921

**7. Cosine Similarity**
Cosine similarity is used to measure the similarity between movie vectors.

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(tag_vectors)

This generates a similarity matrix where each movie can be compared against every other movie. The resulting similarity matrix has dimensions (4804, 4804). fileciteturn0file0L978-L982

**8. Recommendation Function**
The final recommendation function accepts a movie title, finds its index, sorts movies according to similarity, and returns the top 10 similar movies.

def recom(movie):
    index = new_df[new_df["title"] == movie].index[0]
    distances = sorted(
        enumerate(similarity[index]),
        reverse=True,
        key=lambda x: x[1]
    )

    for i in distances[1:11]:
        print(new_df.iloc[i[0]].title)

The selected movie itself is excluded by starting from distances[1:11]. fileciteturn0file0L983-L998

## Example
<img width="819" height="277" alt="image" src="https://github.com/user-attachments/assets/f8a58223-cd72-4807-b588-d6f40b624512" />
<img width="567" height="277" alt="image" src="https://github.com/user-attachments/assets/b0af18b9-09aa-495d-9503-6a23caaa37a9" />
<img width="549" height="298" alt="image" src="https://github.com/user-attachments/assets/4f046a48-e54c-452a-93ab-e0ddfd894c68" />

## Limitations
The system is content-based and does not use individual user ratings or viewing history.
Recommendations depend on the quality and coverage of movie metadata.
The project encountered corrupt/inconsistent records in the credits data, resulting in cast and director information being excluded from the final feature set. 
The recommendation quality is therefore primarily driven by the movie overview, genres, and keywords.

**Future Improvements**
Possible extensions include:
Incorporating cast and director information after robustly cleaning the credits data.
Using TF-IDF instead of raw term counts.
Exploring word embeddings or transformer-based text representations.
Adding user ratings and collaborative filtering.
Building a web interface using Streamlit or Flask.
Adding fuzzy title matching for misspelled or partially entered movie names.

## Author

Manish Kumar

If this project helped you, feel free to ⭐ the repository.
