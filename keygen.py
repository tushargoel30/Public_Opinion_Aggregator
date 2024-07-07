import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def genCloud(keywords_dict, file_path='static/img/wordcloud.png'):
    """
    Generates a word cloud image from a dictionary of keywords and their scores.

    Parameters:
    keywords_dict (dict): A dictionary where keys are words and values are their corresponding scores.
    file_path (str): Path to save the generated word cloud image.
    """
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white')
    wordcloud.generate_from_frequencies(frequencies=keywords_dict)

    # Display the generated image:
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Keyword Word Cloud")
    
    # Save the image to file
    wordcloud.to_file(file_path)
    plt.close()

