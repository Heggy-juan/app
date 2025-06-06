import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings

# 解决Streamlit运行环境问题
if "streamlit" not in sys.modules:
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    from streamlit.web.bootstrap import run
    sys.argv = ["streamlit", "run", sys.argv[0], "--global.developmentMode=false"]
    sys.exit(run())

warnings.filterwarnings("ignore", category=UserWarning)

# 多语言支持
LANGUAGES = {
    "中文": {
        "title": "机器学习回归模型预测平台",
        "upload": "上传Excel数据文件 (例: 团簇+单原子-分开算.xlsx)",
        "decimal": "保留小数位数",
        "model_perf": "模型性能对比",
        "best_model": "当前数据表现最优的模型为",
        "tuning": "自动调参",
        "final_perf": "最终模型在测试集表现",
        "prediction": "输入特征做预测",
        "compound": "化合物名称",
        "paste": "可直接粘贴特征值（以Tab、逗号、空格或回车分隔）",
        "predict_btn": "预测",
        "result": "预测结果",
        "raw_data": "原始数据",
        "upload_info": "请先上传包含'OER过电势'和'ORR过电势'两列的Excel文件",
        "error_columns": "未检测到'OER过电势'和'ORR过电势'两列，请确认数据格式和表头！",
        "error_data": "特征列或目标列为空，请检查数据！",
        "error_paste": "粘贴的数据数量或格式不对，请检查！",
        "features_error": "特征数量错误",
        "no_tuning": "无需调参",
        "download": "下载预测结果",
        "features": "特征数量",
        "samples": "样本数量",
        "data_overview": "数据概览",
        "model_training": "模型训练",
        "prediction_section": "化合物预测",
        "data_analysis": "数据分析",
        "app_intro": "应用介绍",
        "feature_corr": "特征相关性",
        "data_dist": "数据分布",
        "history_pred": "历史预测记录"
    },
    "English": {
        "title": "Machine Learning Regression Prediction Platform",
        "upload": "Upload Excel Data File (e.g. cluster+single_atom.xlsx)",
        "decimal": "Decimal Places",
        "model_perf": "Model Performance Comparison",
        "best_model": "Best Performing Model for Current Data",
        "tuning": "Automatic Hyperparameter Tuning",
        "final_perf": "Final Model Performance on Test Set",
        "prediction": "Input Features for Prediction",
        "compound": "Compound Name",
        "paste": "Paste feature values (separated by Tab, comma, space or Enter)",
        "predict_btn": "Predict",
        "result": "Prediction Result",
        "raw_data": "Raw Data",
        "upload_info": "Please upload an Excel file containing 'OER Overpotential' and 'ORR Overpotential' columns",
        "error_columns": "Required columns 'OER Overpotential' and 'ORR Overpotential' not found. Please check data format and headers!",
        "error_data": "Feature columns or target columns are empty. Please check data!",
        "error_paste": "Pasted data format is incorrect. Please check!",
        "features_error": "Feature count error",
        "no_tuning": "No tuning required",
        "download": "Download Prediction Results",
        "features": "Number of Features",
        "samples": "Number of Samples",
        "data_overview": "Data Overview",
        "model_training": "Model Training",
        "prediction_section": "Compound Prediction",
        "data_analysis": "Data Analysis",
        "app_intro": "Application Introduction",
        "feature_corr": "Feature Correlation",
        "data_dist": "Data Distribution",
        "history_pred": "Prediction History"
    },
    "Español": {
        "title": "Plataforma de Predicción de Modelos de Regresión de Aprendizaje Automático",
        "upload": "Cargar archivo Excel de datos (ej. cluster+single_atom.xlsx)",
        "decimal": "Decimales",
        "model_perf": "Comparación de Rendimiento de Modelos",
        "best_model": "Modelo de Mejor Rendimiento para los Datos Actuales",
        "tuning": "Ajuste Automático de Hiperparámetros",
        "final_perf": "Rendimiento Final del Modelo en Conjunto de Prueba",
        "prediction": "Ingresar Características para Predicción",
        "compound": "Nombre del Compuesto",
        "paste": "Pegar valores de características (separados por Tab, coma, espacio o Enter)",
        "predict_btn": "Predecir",
        "result": "Resultado de Predicción",
        "raw_data": "Datos Crudos",
        "upload_info": "Por favor cargue un archivo Excel que contenga las columnas 'Sobrepotencial OER' y 'Sobrepotencial ORR'",
        "error_columns": "Columnas requeridas 'Sobrepotencial OER' y 'Sobrepotencial ORR' no encontradas. ¡Por favor verifique el formato de datos y encabezados!",
        "error_data": "Las columnas de características o las columnas objetivo están vacías. ¡Por favor verifique los datos!",
        "error_paste": "El formato de los datos pegados es incorrecto. ¡Por favor verifique!",
        "features_error": "Error en el recuento de características",
        "no_tuning": "No se requiere ajuste",
        "download": "Descargar Resultados de Predicción",
        "features": "Número de Características",
        "samples": "Número de Muestras",
        "data_overview": "Resumen de Datos",
        "model_training": "Entrenamiento de Modelo",
        "prediction_section": "Predicción de Compuesto",
        "data_analysis": "Análisis de Datos",
        "app_intro": "Introducción de la Aplicación",
        "feature_corr": "Correlación de Características",
        "data_dist": "Distribución de Datos",
        "history_pred": "Historial de Predicciones"
    },
    "日本語": {
        "title": "機械学習回帰モデル予測プラットフォーム",
        "upload": "Excelデータファイルをアップロード（例: cluster+single_atom.xlsx）",
        "decimal": "小数点以下の桁数",
        "model_perf": "モデル性能比較",
        "best_model": "現在のデータで最適なモデル",
        "tuning": "自動ハイパーパラメータチューニング",
        "final_perf": "テストセットでの最終モデル性能",
        "prediction": "予測のための特徴量入力",
        "compound": "化合物名",
        "paste": "特徴量を貼り付け（タブ、カンマ、スペースまたは改行で区切る）",
        "predict_btn": "予測",
        "result": "予測結果",
        "raw_data": "生データ",
        "upload_info": "'OER過電圧'と'ORR過電圧'の列を含むExcelファイルをアップロードしてください",
        "error_columns": "必要な列'OER過電圧'と'ORR過電圧'が見つかりません。データ形式とヘッダーを確認してください！",
        "error_data": "特徴量列またはターゲット列が空です。データを確認してください！",
        "error_paste": "貼り付けたデータの形式が正しくありません。確認してください！",
        "features_error": "特徴量数エラー",
        "no_tuning": "チューニング不要",
        "download": "予測結果をダウンロード",
        "features": "特徴量の数",
        "samples": "サンプル数",
        "data_overview": "データ概要",
        "model_training": "モデルトレーニング",
        "prediction_section": "化合物予測",
        "data_analysis": "データ分析",
        "app_intro": "アプリケーション紹介",
        "feature_corr": "特徴量相関",
        "data_dist": "データ分布",
        "history_pred": "予測履歴"
    }
}

# 初始化Session State
if 'language' not in st.session_state:
    st.session_state.language = "中文"
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'y_cols' not in st.session_state:
    st.session_state.y_cols = ['OER过电势', 'ORR过电势']

# 语言选择器
def set_language():
    st.session_state.language = st.session_state.lang_select
    # 根据语言更新目标列名
    lang = st.session_state.language
    if lang == "中文":
        st.session_state.y_cols = ['OER过电势', 'ORR过电势']
    elif lang == "English":
        st.session_state.y_cols = ['OER Overpotential', 'ORR Overpotential']
    elif lang == "Español":
        st.session_state.y_cols = ['Sobrepotencial OER', 'Sobrepotencial ORR']
    elif lang == "日本語":
        st.session_state.y_cols = ['OER過電圧', 'ORR過電圧']

# 获取当前语言文本
def t(key):
    return LANGUAGES[st.session_state.language][key]

# 添加自定义CSS样式
def add_custom_css():
    st.markdown("""
    <style>
        /* 主标题样式 */
        .stApp header h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        /* 侧边栏样式 */
        [data-testid="stSidebar"] {
            background: linear-gradient(135deg, #2c3e50, #4a6491);
            color: white;
        }
        
        [data-testid="stSidebar"] .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #2980b9;
        }
        
        /* 卡片样式 */
        .card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* 按钮样式 */
        .stButton button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            background-color: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* 表单样式 */
        .stForm {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* 指标卡样式 */
        .metric {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        /* 页脚样式 */
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            background-color: #2c3e50;
            color: white;
            text-align: center;
            padding: 10px;
            z-index: 100;
        }
    </style>
    """, unsafe_allow_html=True)

# 文件上传与处理
def handle_file_upload():
    uploaded_file = st.file_uploader(t("upload"), type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, header=1)
        except Exception as e:
            st.error(f"Excel文件读取失败：{e}")
            return None
        
        # 自动检测表头行
        if not all(col in df.columns for col in st.session_state.y_cols):
            try:
                df = pd.read_excel(uploaded_file, header=2)
            except:
                pass

        if not all(col in df.columns for col in st.session_state.y_cols):
            st.error(t("error_columns"))
            return None

        compound_col = df.columns[0]
        df = df.dropna(subset=st.session_state.y_cols)
        return df
    
    return None

# 训练模型
def train_models(X, Y):
    # 分割数据集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso Regression": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.1))]),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
    }

    results = {}
    best_score = -np.inf
    best_model_name = None
    best_model_object = None

    # 模型训练和评估
    with st.spinner(t("model_perf")):
        for name, model in models.items():
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            mse = mean_squared_error(Y_test, Y_pred)
            mae = mean_absolute_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)
            cv_score = cross_val_score(model, X, Y, cv=5, scoring='r2').mean()
            results[name] = {
                "MSE": mse,
                "MAE": mae,
                "R²": r2,
                "CV R²": cv_score
            }
            if cv_score > best_score:
                best_score = cv_score
                best_model_name = name
                best_model_object = model

    # 自动调参
    param_grids = {
        "Random Forest": {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [5, 10, None]
        },
        "Gradient Boosting": {
            "estimator__n_estimators": [100, 200],
            "estimator__max_depth": [3, 5, None]
        },
        "Ridge Regression": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0]
        },
        "Lasso Regression": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0]
        }
    }

    if best_model_name in param_grids:
        st.info(f"{t('tuning')} {best_model_name}...")
        if best_model_name == "Random Forest":
            base_estimator = MultiOutputRegressor(RandomForestRegressor(random_state=42))
            param_grid = param_grids["Random Forest"]
        elif best_model_name == "Gradient Boosting":
            base_estimator = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
            param_grid = param_grids["Gradient Boosting"]
        elif best_model_name == "Ridge Regression":
            base_estimator = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            param_grid = param_grids["Ridge Regression"]
        elif best_model_name == "Lasso Regression":
            base_estimator = Pipeline([("scaler", StandardScaler()), ("model", Lasso())])
            param_grid = param_grids["Lasso Regression"]

        grid = GridSearchCV(base_estimator, param_grid, cv=5, n_jobs=-1)
        grid.fit(X_train, Y_train)
        best_params = grid.best_params_
        st.write(f"{best_model_name} {t('best_model')}: {best_params}")
        best_model_object = grid.best_estimator_
    else:
        st.info(f"{best_model_name} {t('no_tuning')}")
    
    return results, best_model_object, best_model_name, best_score, X_test, Y_test

# 主应用
def main():
    add_custom_css()
    
    # 侧边栏
    with st.sidebar:
        st.header("设置")
        st.selectbox(
            "选择语言 / Language", 
            list(LANGUAGES.keys()), 
            key="lang_select", 
            on_change=set_language
        )
        
        decimal_places = st.selectbox(t("decimal"), [1, 2, 3, 4, 5], index=2)
        
        st.markdown("---")
        st.subheader(t("prediction"))
        st.info(t("paste"))
        
        st.markdown("---")
        st.subheader("关于")
        st.markdown("""
        此应用使用多种机器学习回归模型预测化合物的OER和ORR过电势。
        - 支持线性回归、岭回归、Lasso回归、随机森林和梯度提升树
        - 自动选择最佳模型并进行超参数调优
        - 提供批量特征值预测功能
        """)
        
        st.markdown("---")
        st.caption("© 2023 机器学习预测平台 | 版本 1.2.0")
    
    # 主界面
    st.title(t("title"))
    
    # 文件上传与处理
    if st.session_state.df is None:
        st.session_state.df = handle_file_upload()
    
    # 数据概览
    if st.session_state.df is not None:
        df = st.session_state.df
        compound_col = df.columns[0]
        X = df.drop([compound_col] + st.session_state.y_cols, axis=1).select_dtypes(include='number')
        Y = df[st.session_state.y_cols].astype(float)
        
        if X.empty or Y.empty:
            st.error(t("error_data"))
            st.stop()
        
        # 数据概览卡片
        st.subheader(t("data_overview"))
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(t("samples"), len(df))
        with col2:
            st.metric(t("features"), len(X.columns))
        with col3:
            st.metric("目标变量", len(st.session_state.y_cols))
        
        # 模型训练
        st.subheader(t("model_training"))
        if st.button("开始训练模型") or st.session_state.best_model is not None:
            if st.session_state.best_model is None:
                results, best_model, best_model_name, best_score, X_test, Y_test = train_models(X, Y)
                st.session_state.best_model = best_model
                st.session_state.results = results
                st.session_state.best_model_name = best_model_name
                st.session_state.best_score = best_score
                st.session_state.X_test = X_test
                st.session_state.Y_test = Y_test
            
            # 显示模型性能对比
            results_df = pd.DataFrame(st.session_state.results).T
            st.subheader(t("model_perf"))
            
            # 格式化显示
            formatted_results = results_df.copy()
            for col in formatted_results.columns:
                formatted_results[col] = formatted_results[col].apply(lambda x: f"{x:.{decimal_places}f}")
            
            st.dataframe(formatted_results.sort_values(by="CV R²", ascending=False))
            
            st.success(f"{t('best_model')}：{st.session_state.best_model_name}（CV R² = {st.session_state.best_score:.{decimal_places}f}）")
            
            # 测试集最终表现
            Y_pred = st.session_state.best_model.predict(st.session_state.X_test)
            st.subheader(t("final_perf"))
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("MSE", f"{mean_squared_error(st.session_state.Y_test, Y_pred):.{decimal_places}f}")
            with col2:
                st.metric("MAE", f"{mean_absolute_error(st.session_state.Y_test, Y_pred):.{decimal_places}f}")
            with col3:
                st.metric("R²", f"{r2_score(st.session_state.Y_test, Y_pred):.{decimal_places}f}")
            
            # 预测表单
            st.subheader(t("prediction_section"))
            with st.form("predict_form"):
                compound_name = st.text_input(t("compound"), "新化合物")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(t("paste"))
                with col2:
                    example_features = ", ".join([f"{np.random.uniform(0, 1):.3f}" for _ in range(len(X.columns))])
                    st.caption(f"示例: {example_features}")
                
                # 修复高度问题
                pasted_features = st.text_area("", value="", height=70, label_visibility="collapsed")
    
                feature_values = []
                valid_paste = False
                if pasted_features.strip():
                    raw = pasted_features.replace('\n', ' ').replace('\r', ' ')
                    for sep in ['\t', ',', ' ']:
                        parts = [x for x in raw.split(sep) if x.strip() != ""]
                        if len(parts) == len(X.columns):
                            try:
                                feature_values = [float(x) for x in parts]
                                valid_paste = True
                                break
                            except:
                                pass
                    if not valid_paste:
                        st.error(f"{t('error_paste')}（需要 {len(X.columns)} 个特征）")
    
                if not valid_paste:
                    feature_values = [float(X[col].mean()) for col in X.columns]
    
                # 特征输入框
                cols = st.columns(3)
                col_idx = 0
                for i, col in enumerate(X.columns):
                    with cols[col_idx]:
                        feature_values[i] = st.number_input(f"{col}", value=feature_values[i], step=0.01)
                    col_idx = (col_idx + 1) % 3
                
                # 添加提交按钮
                submit_btn = st.form_submit_button(t("predict_btn"))
            
            # 处理预测请求
            if submit_btn:
                if len(feature_values) != X.shape[1]:
                    st.error(f"{t('features_error')}，需要 {X.shape[1]} 个。")
                else:
                    input_df = pd.DataFrame([feature_values], columns=X.columns)
                    prediction = st.session_state.best_model.predict(input_df)
                    
                    # 保存预测结果
                    result = {
                        "Compound": compound_name,
                        "Features": feature_values.copy(),
                        "OER": prediction[0][0],
                        "ORR": prediction[0][1]
                    }
                    st.session_state.prediction_results.append(result)
                    
                    # 显示预测结果
                    st.success(f"{compound_name} {t('result')}：")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label=st.session_state.y_cols[0], 
                            value=f"{prediction[0][0]:.{decimal_places}f} V",
                            delta=None
                        )
                    with col2:
                        st.metric(
                            label=st.session_state.y_cols[1], 
                            value=f"{prediction[0][1]:.{decimal_places}f} V",
                            delta=None
                        )
            
            # 显示历史预测结果
            if st.session_state.prediction_results:
                st.subheader(t("history_pred"))
                history_df = pd.DataFrame(st.session_state.prediction_results)
                history_display = history_df[["Compound", "OER", "ORR"]]
                st.dataframe(history_display.style.format({
                    'OER': f"{{:.{decimal_places}f}}",
                    'ORR': f"{{:.{decimal_places}f}}"
                }))
                
                # 下载预测结果
                csv = history_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=t("download"),
                    data=csv,
                    file_name='prediction_results.csv',
                    mime='text/csv'
                )
            
            # 数据分析
            st.subheader(t("data_analysis"))
            
            # 数据可视化
            tab1, tab2 = st.tabs([t("data_dist"), t("feature_corr")])
            
            with tab1:
                st.subheader(t("data_dist"))
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(df[st.session_state.y_cols[0]].value_counts())
                with col2:
                    st.bar_chart(df[st.session_state.y_cols[1]].value_counts())
            
            with tab2:
                st.subheader(t("feature_corr"))
                corr = X.corr()
                st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None).format("{:.2f}"))
            
            # 原始数据展示
            with st.expander(f"📊 {t('raw_data')}"):
                st.dataframe(df)
    else:
        # 应用介绍
        st.info(t("upload_info"))
        
        # 应用功能介绍
        with st.expander(t("app_intro")):
            st.markdown(f"""
            ### {t('app_intro')}
            
            **{t('title')}** 是一个用于预测化合物OER和ORR过电势的工具，具有以下功能：
            
            1. **多语言支持**：
               - 支持中文、英文、西班牙文和日文界面
               - 实时切换语言
            
            2. **模型训练与评估**：
               - 支持多种机器学习回归模型（线性回归、岭回归、Lasso回归、随机森林、梯度提升树）
               - 自动评估模型性能并选择最佳模型
               - 提供详细的模型性能指标（MSE、MAE、R²、CV R²）
            
            3. **自动超参数调优**：
               - 对选择的模型自动进行超参数优化
               - 显示最优参数组合
            
            4. **预测功能**：
               - 支持单次预测
               - 批量粘贴特征值进行预测
               - 保存历史预测记录
               - 可视化预测结果
            
            5. **数据分析**：
               - 数据分布可视化
               - 特征相关性分析
               - 原始数据展示
            
            ### 使用说明
            
            1. **上传数据**：
               - 上传包含'OER过电势'和'ORR过电势'两列的Excel文件
               - 确保数据格式正确（第一列为化合物名称，后续列为特征值）
            
            2. **模型训练**：
               - 点击"开始训练模型"按钮
               - 系统会自动训练多个模型并评估性能
               - 选择最佳模型并进行超参数优化
            
            3. **预测化合物**：
               - 输入化合物名称
               - 粘贴或输入特征值
               - 点击"预测"按钮查看结果
            
            4. **结果分析**：
               - 查看预测结果（OER和ORR过电势）
               - 分析历史预测记录
               - 下载预测结果
            """)
    
    # 页脚 - 居中显示
    st.markdown("""
    <div class="footer">
        <p style="text-align: center; width: 100%;">机器学习预测平台 © 2023 | 技术支持: support@mlprediction.com</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()