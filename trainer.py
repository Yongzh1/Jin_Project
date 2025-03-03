import json
import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from evaluator import Evaluator
from imblearn.under_sampling import RandomUnderSampler
from optuna.integration import OptunaSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model_name,
        if_under_sampling,
        if_fill_missing,
        window_size,
        config_path="Config/params.yaml",
        n_trials=200,
    ):
        self.model_name = model_name
        self.file_name = f"{model_name}_ws_{window_size}"
        self.model = None
        self.best_params = None
        self.param_config = self.load_yaml(config_path)
        self.load_model()
        self.n_trials = n_trials
        self.us_flag = if_under_sampling
        self.save_params_path = "Models/params/"

    def load_yaml(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config["parameters"].get(self.model_name, {})

    def load_model(self):
        """
        モデルを選択する
        """
        if self.model_name == "svm":
            from sklearn.svm import SVC

            self.model = SVC(probability=True, kernel="linear", random_state=42)
        elif self.model_name == "lgbm":
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(verbose=-1, random_seed=42)
        elif self.model_name == "xgboost":
            from xgboost import XGBClassifier

            self.model = XGBClassifier(eval_metric="logloss", random_state=42)
        elif self.model_name == "catboost":
            from catboost import CatBoostClassifier

            self.model = CatBoostClassifier(verbose=0, random_seed=42)
        elif self.model_name == "lr":
            from sklearn.linear_model import LogisticRegression

            self.model = LogisticRegression(random_state=42)
        else:
            print("モデルが対応していません")
        # print(f"{self.model_name} model loaded")

    def objective(self, trial, X, y, groups, n_splits=5):
        """
        Optuna の目的関数 (ハイパーパラメータの最適化を行う)
        """

        # ハイパーパラメータ
        params = {}
        for param_name, param_info in self.param_config.items():
            if param_info["type"] == "int":
                params[param_name] = trial.suggest_int(
                    param_name, param_info["low"], param_info["high"]
                )
            elif param_info["type"] == "float":
                params[param_name] = trial.suggest_float(
                    param_name, param_info["low"], param_info["high"]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_info['type']}")

        # GroupKFold で CV を実施
        scores = []
        for train_idx, valid_idx in GroupKFold(n_splits=n_splits).split(
            X, y, groups=groups
        ):

            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            # UnderSampling を適用
            if self.us_flag == True:
                rus = RandomUnderSampler(random_state=42)
                X_train_resampled, y_train_resampled = rus.fit_resample(
                    X_train, y_train
                )
            else:  # flagがFalseだとそのまま
                X_train_resampled = X_train
                y_train_resampled = y_train

            # インスタンス内モデルの初期化
            self.load_model()

            # モデルにハイパーパラメータを設定
            self.model.set_params(**params)

            # モデルの学習
            self.fit(X_train_resampled, y_train_resampled)

            # 検証データで予測
            y_pred = self.predict(X_valid)

            # f1スコアを計算
            score = f1_score(y_valid, y_pred)
            scores.append(score)

        # 平均 F1 スコアを返す
        return np.mean(scores)

    def optimize(self, X, y, groups, n_splits=5):
        # OptunaのログレベルをWARNINGに設定（詳細なINFOログを非表示にする）
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # ランダムシードを固定したSamplerを作成
        sampler = optuna.samplers.RandomSampler(seed=42)  # TPE Samplerを使用

        # OptunaのStudyを作成
        n_trials = self.n_trials
        study = optuna.create_study(
            direction="maximize", sampler=sampler
        )  # f1スコアを最大化
        with tqdm(
            total=self.n_trials, desc="Hyperparameter Tuning", unit="trial"
        ) as pbar:

            def objective_with_progress(trial):
                score = self.objective(trial, X, y, groups, n_splits)
                pbar.update(1)  # 1 trial 終わるごとに進捗バーを更新
                return score

            study.optimize(objective_with_progress, n_trials=self.n_trials)

        # 最適なパラメータとスコアを取得
        self.best_params = study.best_params
        #print(f"Best parameters: {self.best_params}")

        # 最適なパラメータをファイルとして保存
        output_file = "Models/params/" + self.file_name + ".json"  # 保存するファイル名
        with open(output_file, "w") as f:
            json.dump(self.best_params, f, indent=4)

        print(f"Best parameters saved to {output_file}")

    def retrain_model(self, X, y):
        # インスタンス内モデルの初期化
        self.load_model()

        # 最適なパラメータをモデルにセットして再学習
        self.model.set_params(**self.best_params)

        # UnderSampling を適用
        if self.us_flag == True:
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
        else:  # flagがFalseだとそのまま
            X_resampled = X
            y_resampled = y

        self.fit(X_resampled, y_resampled)  # モデルを学習データで再学習

        print("Model re-trained with best parameters.")

        # モデルを保存
        model_filename = f"{self.file_name}.joblib"
        joblib.dump(
            self.model, "Models/trained_model/" + model_filename
        )  # モデルをファイルに保存
        print(f"Model saved as {model_filename}")

    def fit(self, X_train, y_train):
        # 学習
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # 予測
        y_pred = self.model.predict(X_test)
        return y_pred
