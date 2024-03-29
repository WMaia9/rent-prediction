Real Estate São Paulo Rent Prediction 
-----------------------

Predicts the value of rental properties in the city of São Paulo, Brazil, using data extracted from the Zap Imoveis website and adding new variable.

Installation
----------------------
### Install the requirements
 
* Install the requirements using `pip install -r requirements.txt`.
    * Make sure you use Python 3.
    * You may want to use a virtual environment for this.
  
### Download the data

* Clone this repo to your computer.
* Get into the folder using `cd rest-prediction`.
* Run `webcr.py` to make a web scraper and pull the data from Zap Imóveis.  
    * You can find where the data was extracted from [here](https://www.zapimoveis.com.br/).
    * It's recommended to choose a least 2 pages in `webcr.py`.
 * Switch into the `data` directory using `cd data` to see the data ready.



Usage
-----------------------

* Run `python assemble.py` to combine the `sp+sao-paulo+centro`,  `sp+sao-paulo+zona-leste`,  `sp+sao-paulo+zona-norte`, `sp+sao-paulo+zona-oest` and `sp+sao-paulo+zona-sul` datasets.
    * This will create `sp.csv` in the `processed` folder.
* Run `python annotate.py`.
    * This will create `geo.csv` data from `sp.csv`.
* Run `python geotransformation.py`.
    * This will create training and Test data from `geo.csv`.
    * It will add a file called `train.csv` and `test.csv`to the `processed` folder.
* Run `python predict.py`.
    * This will run cross validation across the training and test set, and print the rmse score.
  
* Additional Data:
    * In the sp folder you can find the data used to add new variables to our data.
    * The folder also contains the final train and test data with more than 30,000 data used by me to run the predict algorithm.
    * `python geofast.py` is used to calculate geospacial distances.
  
Extending this
-------------------------

If you want to extend this work, Here few places to start:

* Generate more features in `annotate.py`.
* Switch algorithms in `predict.py`.
* Add in a way to make predictions on future data.
    * Make predictions.
* Explore seeing if you can predict columns other than `price`.
* Explore the nuances between performance updates.
