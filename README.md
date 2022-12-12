# MOVIE-BOX-OFFICE-PREDICTION-USING-A-SELF-SUPERVISED-LEARNING-TRANSFORMER-ENCODER

We introduce a new, large-scale, multimodal, text-image paired, movie meta-data dataset here. 
The dataset contains 35,794 movies with its meta data and the movie poster crawled from www.tmdb.com. The box office information is crawled from www.imdbpro.com. 
Each movie is assigned an unique tmdb id, e.g., the tmdb id for movie `titanic’ is 597. https://www.themoviedb.org/movie/597-titanic. The tmdb_id is used to retrieval the information. 
 
### Meta data: basics, keywords, and credits (Total 202 MB)
There are folders contains movies’ basic information, keywords and credits, respectively. Files are named as **<tmdb_id>.npy**
File Load Example:
``` 
>>data=np.load('../credits/5.npy', allow_pickle=True)
>>print(type(data))
<class 'dict'>
```
Click [here](https://drive.google.com/file/d/10g05aoMeClUxEjMFSt_5A5xjYPWI8JlK/view?usp=sharing) to download.

### Box Office (.csv)
We crawled movie’s box office information from https://pro.imdb.com/, named as **IMDB_crawl_36k_rawdata.csv**

### Poster, image size: w400*h600 (Total 1.5 GB)
Click [here] (https://drive.google.com/file/d/1AfsflHjQSuFPPZB7qfpO4-dcR45bIeNa/view?usp=sharing) to download.
Files are named as **<tmdb_id>.jpg**

### Poster High-level features (Total 36 GB)
We produced and released high-level (feature after ROI Align) features files, named as named as **<tmdb_id>.npy**. 
under folder **text.txt**.
A bbox area, the confidence score and the object name. 
Output example:
 ```
>>path='../poster_features/'+str(int(597))+'.npy'
>>arr=np.load(path)
>>image_feature_maps = torch.from_numpy(arr)
>>print(image_feature_maps.shape)
torch.Size([29, 2048, 7, 7])
```
```
>>path = '../attr_bbox/'+str(int(597))+'.txt
>>f = open(path, 'r')
>>data = f.readlines()
>>print(data)
['gold,silver,earring\n', 
'0.1127,0.4715\n',  #prediction confidence score for attribute “gold” and “silver”, respectively
'0.9832\n', #confidence score for object “earring”
'[231.6, 123.3, 244.4, 164.3]\n', # bbox location, mode = [x1, yi, x2, y2]
…
```

According to the bbox prediction. In the same order
Click [here](google drive share link) to download.
