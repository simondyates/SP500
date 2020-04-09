# SP500

The main file in this project is FactorModel.py which decomposes the stock returns (loaded from SP500.csv) into factor weightings on ETFs (from ETF.csv)
The process used to identify the most relevant ETFs, and a parsimonious selection of them is forward stepwise regression.  At each stage, the selected ETF is used as a basis vector in a Gram Schmidt orthonormalisation process.  Consequently, the resulting set of vectors is orthonormal.  The selection of basis vectors stops once a predetermined threshold for the average t-stat is reached. (In the current version of the code this is set to 1.5).

Typical applications of factor decomposition include understanding the macro risks associated with a stock portfolio, and/or determining the best hedge trades to reduce these exposures.

The file SPXWts_31Mar2020.csv was used to determine the population of stocks that make up the index.
