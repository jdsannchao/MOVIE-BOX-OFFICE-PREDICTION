# MOVIE-BOX-OFFICE-PREDICTION-USING-A-SELF-SUPERVISED-LEARNING-TRANSFORMER-ENCODER

We introduce a new, large-scale, multimodal, text-image paired, movie meta-data dataset. 
The dataset contains 35,794 movies with its meta data and movie poster crawled from www.tmdb.com. We also include movie's theatrical performance,  such as box office, and number of screens; source came from www.imdbpro.com. 
Each movie has been assigned an unique tmdb id, e.g., the tmdb id for movie [Titanic](https://www.themoviedb.org/movie/597-titanic) is 597. The tmdb_id is used during information retrieval. 
 
To access the data, please fill out the [form](https://docs.google.com/forms/d/e/1FAIpQLSfB_9j_8fYF6YpjEBrUecxs-VNAFrmNxKD2zPQYQSXCcIUbKA/viewform?usp=sf_link) 
 
### Meta data: basics, keywords, and credits (Total 202 MB)
There are folders contains movies’ basic information, keywords and credits, respectively. Files are named as ****<tmdb_id>.npy****

Data Loading Example:
``` 
>>data=np.load('../credits/5.npy', allow_pickle=True)
>>print(type(data))
<class 'dict'>
```

### Box Office (.csv)
We crawled movie’s box office information from https://pro.imdb.com/, named as **IMDB_crawl_36k_rawdata.csv**

### Poster (Image size: w400*h600, Total 1.5 GB)
Files are named as **<tmdb_id>.jpg**

### Poster High-level features (Total 36 GB)
We produced and released high-level features files, zipped as **ROI_Align_features.zip**. Feature for each movie is named as **<tmdb_id>.npy**. 

Output example:
 ```
>>path='../features_2048_4_4/'+str(int(597))+'.npy'
>>arr=np.load(path)
>>image_feature_maps = torch.from_numpy(arr)
>>print(image_feature_maps.shape)
torch.Size([29, 2048, 4, 4])  #number of objects detected, channels, h, w 
```

The feature map collate with the bbox prediction result which is named as **attr_bbox.zip**. Inside the folder, each movie has its **<tmdb_id>.txt** listing the detected bbox area with the confidence score and the object class.

Output example:
```
>>path = '../attr_bbox/'+str(int(597))+'.txt
>>f = open(path, 'r')
>>data = f.readlines()
>>print(data)
['gold,silver,earring\n', ## this area is predicted as "earring"
'0.1127,0.4715\n',  ## the "earring" is predicted to be "gold" and "silver" with the confidence score of "0.11" and "0.47", respectively
'0.9832\n', ## confidence score for object "earring"
'[231.6, 123.3, 244.4, 164.3]\n', # bbox location, mode = [x1, y1, x2, y2]
…
```
The feature map and the text file share the same order. 

