from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from index import load_accident_data, load_traffic_flow_data, preprocess_data, combine_data, run_dbscan, run_kmeans, \
    get_top_density_clusters, find_cluster_names, plot_top_clusters



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    algorithm = request.form['algorithm']
    accidents_data = load_accident_data()
    traffic_flow_data = load_traffic_flow_data()
    accidents_data, traffic_flow_data = preprocess_data(accidents_data, traffic_flow_data)
    combined_data = combine_data(accidents_data, traffic_flow_data)

    if algorithm == 'dbscan':
        labels = run_dbscan(combined_data)
    elif algorithm == 'kmeans':
        labels = run_kmeans(combined_data)
    else:
        labels = None  # Handle invalid algorithm selection

    top_clusters = get_top_density_clusters(combined_data, labels)
    cluster_names = find_cluster_names(labels)

    plot_data = None
    kmeans_plot_data = None

    # Generate the scatter plot
    if algorithm in ['dbscan', 'kmeans'] and labels is not None:
        plt.figure(figsize=(4, 4))
        plt.scatter(combined_data['Longitude'], combined_data['Latitude'], c=labels, cmap='viridis',
                    s=combined_data['counts'] / combined_data['weight'] * 10)
        plt.title('Scatter Plot of Locations with Clusters')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar(label='Cluster')

        # Convert plot to base64 encoded image
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

    if algorithm in ['kmeans', 'dbscan'] and labels is not None:
        kmeans_plot_data = plot_top_clusters(accidents_data, labels, top_clusters)

    # Define the results variable
    results = {
        'algorithm': algorithm,
        'labels': labels,
        'top_clusters': top_clusters,
        'cluster_names': cluster_names,
        'plot_data': plot_data,
        'kmeans_plot_data': kmeans_plot_data  # Add plot data for second plot
    }

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
