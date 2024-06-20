from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_color = {}
        self.player_team_dict = {}
    
    def get_clustering_kmeans(self, image):
        # Reshape image to 2D array
        img_2d = image.reshape(-1, 3)
        # perform Kmean with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++", n_init=1).fit(img_2d)
        
        return kmeans        
    
    def get_player_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = img[0:int(img.shape[0]/2), :]
        
        # Get clustering kmean of top half of image
        kmeans = self.get_clustering_kmeans(top_half_image)
        
        # Get the cluster labels for each pixel 
        labels = kmeans.labels_
        # Reshape the labels to image shape
        cluster_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])
        
        # Get player cluster
        corner_clusters = [cluster_image[0,0], cluster_image[0,-1], cluster_image[-1,0], cluster_image[-1,-1]]
        non_player_clusters = max(set(corner_clusters), key=corner_clusters.count)
        player_clusters = 1 - non_player_clusters
        
        player_color = kmeans.cluster_centers_[player_clusters]
        
        return player_color
        
    
    def assign_team_color(self, frame, player_detection):
        
        player_colors = []
        for _, player_detection in player_detection.items():
            bbox = player_detection['bbox']
            player_colors.append(self.get_player_color(frame, bbox))
            
        kmeans = KMeans(n_clusters=2, random_state=0, init="k-means++", n_init=10).fit(player_colors)
        
        self.kmeans = kmeans
        
        self.team_color[1] = kmeans.cluster_centers_[0]
        self.team_color[2] = kmeans.cluster_centers_[1]
        
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # Get player color
        player_color = self.get_player_color(frame, player_bbox)
        
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1
        
        if player_id == 70 or player_id == 102:
            team_id = 2
            
        self.player_team_dict[player_id] = team_id
        
        return team_id
    
        
            