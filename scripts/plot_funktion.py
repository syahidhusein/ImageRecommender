import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# plot function for dimension reduction
def reduction_plot(embedding_reduced, labels, title=None):
    df_plot = pd.DataFrame({'x': embedding_reduced[:, 0], 'y': embedding_reduced[:, 1], 'label': labels["label"]})
    fig = px.scatter(df_plot, x='x', y='y', color='label', opacity=0.3, hover_name=labels["image_path"])
    
    fig.update_traces(marker=dict(line=dict(width=0.3, color='DarkSlateGrey')), selector=dict(mode='markers'))
    fig.update_layout(width=1000, height=800, title=title, title_font=dict(size=24))
    fig.show()

# Create plots for the top 8 most similar images relative to a subject image
def images_grid(img_path, sim_score, title=None):
    num_images = len(img_path)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_images):
        image = cv2.imread(img_path[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].text(0, 0, f'score: {sim_score[i]:6f}', color='white', fontsize=12, 
                     bbox=dict(facecolor='black', edgecolor='white', pad=3.0))

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.show()

# Create plots for the top 8 most similar images as histogram
def images_histogram_grid(img_path, sim_score, H=True, S=True, V=True):
    num_images = len(img_path)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()

    for i in range(num_images):
        image = cv2.imread(img_path[i])
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if H:
            axes[i].hist(hsv_image[:, :, 0].ravel(), bins=256, color='r', alpha=0.5, label='Hue', density=True)
        if S:
            axes[i].hist(hsv_image[:, :, 1].ravel(), bins=256, color='g', alpha=0.5, label='Saturation', density=True)
        if V:
            axes[i].hist(hsv_image[:, :, 2].ravel(), bins=256, color='b', alpha=0.5, label='Value', density=True)
        
        axes[i].set_title(f'Score: {sim_score[i]:.6f}')
        axes[i].legend(loc='upper right')
        axes[i].set_xlim([0, 256])
        axes[i].set_xlabel('Bin')
        axes[i].set_ylabel('Normalized Frequency')
    
    plt.tight_layout()
    plt.show()

# Lists the top 8 similar images with their paths
def top_8_gen(img_path, sim_score):
    top_8_paths = img_path['image_path'].head(8).tolist()
    top_8_scores = sim_score['similarity_measure'].head(8).tolist()
    return top_8_paths, top_8_scores