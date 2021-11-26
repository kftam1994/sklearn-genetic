import pandas as pd
import numpy as np
from io import BytesIO
import pmdarima as pm
from pmdarima import model_selection
from genetic_selection import GeneticSelectionPmdarimaCV

friedman2_path = r'Downloads\SARIMAX\friedman2.dta'
friedman2_file = open(friedman2_path,mode='rb')
data = pd.read_stata(BytesIO(friedman2_file.read()))
data.index = data.time

raw = data.loc['1959':'1981', :]

# Variables
Y_raw = raw.loc[:,['consump']]
#X_raw = raw.loc[:,['m2']]
X_raw = raw.loc[:,~raw.columns.isin(['consump','time'])]

#########################################

#mod = pm.auto_arima(y=Y_raw,X=X_raw,information_criterion='aicc',trace=True,random=True,random_state=55)
new_auto_arima_params={'information_criterion':'aicc','trace':True,'random':True,'random_state':55}
estimator = pm.arima.AutoARIMA()
auto_arima_params = estimator.get_params()
auto_arima_params.update(new_auto_arima_params)
estimator = estimator.set_params(**auto_arima_params)
mod = estimator.fit(y=Y_raw,X=X_raw).model_
mod.summary()
mod.plot_diagnostics()

cv = model_selection.RollingForecastCV(step=4, h=1)



mod_cv_scores = model_selection.cross_val_score(estimator=mod, X=X_raw, y=Y_raw , scoring='mean_squared_error', cv=cv, verbose=2)
average_error = np.average(mod_cv_scores)
print(average_error)
#117.0487185350244

########################################

new_auto_arima_params={'information_criterion':'aicc','trace':True,'random':True,'random_state':55}
estimator = pm.arima.AutoARIMA()
auto_arima_params = estimator.get_params()
auto_arima_params.update(new_auto_arima_params)
estimator = estimator.set_params(**auto_arima_params)
cv = model_selection.RollingForecastCV(step=4, h=1)

selector = GeneticSelectionPmdarimaCV(
                            estimator, cv=cv, verbose=1,
                            scoring="mean_squared_error", max_features=X_raw.shape[1],
                            n_population=100, crossover_proba=0.5,
                            mutation_proba=0.2, n_generations=50,
                            crossover_independent_proba=0.5,
                            mutation_independent_proba=0.04,
                            tournament_size=3, n_gen_no_change=10,
                            caching=True, n_jobs=1)
selector = selector.fit(X=X_raw, y=Y_raw)
#122.96869095972544
print('Features:', selector.support_)
# TODO: same model but different coefficients
selector.estimator_.summary()
selector.estimator_.plot_diagnostics()
