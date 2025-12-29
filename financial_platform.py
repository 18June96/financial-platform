from datetime import datetime, date, timedelta
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 导入实训3
from 实训3 import display_financial_analysis, perform_financial_analysis
SHIXUN3_AVAILABLE = True

# 日期转换
def date_to_int(date_obj):
    """将datetime.date对象转换为整数格式的日期（如2024-12-04 → 20241204）"""
    return int(date_obj.strftime("%Y%m%d"))

# 添加六个月
def add_six_months(start_date):
    """计算开始日期加6个月的日期"""
    month = start_date.month + 6
    year = start_date.year + month // 12
    month = month % 12
    if month == 0:
        month = 12
        year -= 1
    try:
        return date(year, month, start_date.day)
    except ValueError:
        next_month = month + 1 if month < 12 else 1
        next_year = year if month < 12 else year + 1
        return date(next_year, next_month, 1) - timedelta(days=1)

# 读取交易数据 CSV文件
t2023 = pd.read_csv('复权交易数据2023.csv')
t2024 = pd.read_csv('复权交易数据2024.csv')
t2025 = pd.read_csv('复权交易数据2025.csv')
trade_data = pd.concat([t2023, t2024, t2025], axis=0)
    
# 确保trade_date是整数类型
trade_data['trade_date'] = trade_data['trade_date'].astype(int)

def calculate_dragon_tiger(start_date_int, end_date_int, trade_data, stock_info_path="股票基本信息表.xlsx"):
    """
    计算龙虎榜数据
    """
    # 1. 读取上市公司基本信息
    stock_info = pd.read_excel(stock_info_path)
    stock_info = stock_info.rename(columns={'name': 'name'})
    
    # 2. 筛选统计区间内的交易数据
    trade_interval = trade_data[
        (trade_data["trade_date"] >= start_date_int) & 
        (trade_data["trade_date"] <= end_date_int)
    ].copy()
    
    if trade_interval.empty:
        st.warning("在选定区间内没有交易数据")
        return pd.DataFrame(columns=["股票代码", "股票简称", "交易所", "涨幅（%）"]), \
               pd.DataFrame(columns=["股票代码", "股票简称", "交易所", "跌幅（%）"])
    
    # 3. 数据清洗-移除价格异常的数据
    trade_interval = trade_interval[trade_interval['close'] > 0]
    
    # 如果pct_chg存在，进行严格过滤
    if 'pct_chg' in trade_interval.columns:
        # A股正常涨跌幅是±10%，新股可能到±44%，这里设置宽松一些
        trade_interval = trade_interval[
            (trade_interval['pct_chg'].between(-30, 30)) | 
            (trade_interval['pct_chg'].isna())
        ]
    
    # 4. 使用两种方法计算并比较
    stock_results = []
    
    for ts_code, group in trade_interval.groupby("ts_code"):
        group = group.sort_values("trade_date")
        
        if len(group) < 5:  # 至少需要5个交易日
            continue
        
        # 方法A：使用价格计算
        first_close = group.iloc[0]["close"]
        last_close = group.iloc[-1]["close"]
        
        if pd.isna(first_close) or first_close <= 0:
            continue
        
        pct_from_price = (last_close - first_close) / first_close * 100
        
        # 方法B：使用pct_chg计算
        pct_from_daily = None
        if 'pct_chg' in group.columns:
            valid_pct = group['pct_chg'].dropna()
            if len(valid_pct) > 0:
                cumulative = 1.0
                for pct in valid_pct:
                    if -20 <= pct <= 20:  # 严格过滤
                        cumulative *= (1 + pct/100)
                pct_from_daily = (cumulative - 1) * 100
        
        # 选择最终结果
        if pct_from_daily is not None:
            # 如果两种方法差异不大，使用pct_chg的结果
            if abs(pct_from_price - pct_from_daily) < 50:  # 差异小于50%
                final_pct = pct_from_daily
            else:
                # 差异大时，取较小值
                final_pct = min(pct_from_price, pct_from_daily) if pct_from_price > 0 else max(pct_from_price, pct_from_daily)
        else:
            final_pct = pct_from_price
        
        # 最终过滤，计算天数
        start_date = datetime.strptime(str(group.iloc[0]["trade_date"]), "%Y%m%d")
        end_date = datetime.strptime(str(group.iloc[-1]["trade_date"]), "%Y%m%d")
        days_diff = (end_date - start_date).days
        
        # 根据天数设置合理的最大涨跌幅
        max_reasonable_pct = min(100, days_diff * 3)  # 每天最多3%
        
        if abs(final_pct) > max_reasonable_pct:
            continue
        
        stock_results.append({
            "ts_code": ts_code,
            "累计涨跌幅(%)": round(final_pct, 2),
            "交易日数": len(group)
        })
    
    if not stock_results:
        st.info("没有计算到有效的涨跌幅数据")
        return pd.DataFrame(columns=["股票代码", "股票简称", "交易所", "涨幅（%）"]), \
               pd.DataFrame(columns=["股票代码", "股票简称", "交易所", "跌幅（%）"])
    
    # 5. 创建结果DataFrame
    stock_pct = pd.DataFrame(stock_results)
    
    # 6. 关联股票基本信息
    stock_pct = pd.merge(stock_pct, stock_info[["ts_code", "name"]], on="ts_code", how="left")
    stock_pct['name'] = stock_pct['name'].fillna(stock_pct['ts_code'])
    
    # 7. 提取交易所信息
    def get_exchange(ts_code):
        if isinstance(ts_code, str):
            if ts_code.endswith(".SH"):
                return "上交所"
            elif ts_code.endswith(".SZ"):
                return "深交所"
            elif ts_code.endswith(".BJ"):
                return "北交所"
        return "未知"
    
    stock_pct["交易所"] = stock_pct["ts_code"].apply(get_exchange)
    
    # 8. 调试信息
    if stock_pct.shape[0] > 0:
        st.info(f"涨跌幅范围：{stock_pct['累计涨跌幅(%)'].min():.2f}% 至 {stock_pct['累计涨跌幅(%)'].max():.2f}%")
    
    # 9. 筛选龙虎榜股票
    up_20 = stock_pct[stock_pct["累计涨跌幅(%)"] > 20].copy()
    down_20 = stock_pct[stock_pct["累计涨跌幅(%)"] < -20].copy()
    
    # 10. 格式化输出
    up_20 = up_20[[
        "ts_code", "name", "交易所", "累计涨跌幅(%)"
    ]].rename(columns={
        "ts_code": "股票代码",
        "name": "股票简称",
        "累计涨跌幅(%)": "涨幅（%）"
    }).sort_values("涨幅（%）", ascending=False).reset_index(drop=True)
    
    down_20 = down_20[[
        "ts_code", "name", "交易所", "累计涨跌幅(%)"
    ]].rename(columns={
        "ts_code": "股票代码",
        "name": "股票简称",
        "累计涨跌幅(%)": "跌幅（%）"
    }).sort_values("跌幅（%）").reset_index(drop=True)
    
    return up_20, down_20
               
def plot_stock_index_charts_actual(start_date_int, end_date_int, trade_data, industry_file='最新个股申万行业分类(完整版-截至7月末).xlsx'):
    """
    使用实际股票数据绘制主要股票价格指数走势图
    1. 上证指数：所有上交所(.SH)股票的平均
    2. 深证指数：所有深交所(.SZ)股票的平均  
    3. 沪深300指数：实际数据
    """
    # 1. 读取行业分类数据
    industry_info = pd.read_excel(industry_file)
    
    # 2. 读取沪深300指数数据
    hs300_data = pd.read_excel('沪深300指数交易数据.xlsx')
    
    # 3. 数据预处理
    # 行业数据：提取股票代码和交易所
    industry_info = industry_info.rename(columns={'股票代码': 'ts_code'})
    
    # 分离上交所和深交所股票
    sh_stocks = industry_info[industry_info['ts_code'].str.endswith('.SH')]['ts_code'].unique()
    sz_stocks = industry_info[industry_info['ts_code'].str.endswith('.SZ')]['ts_code'].unique()
    
    # 沪深300数据：处理日期
    hs300_data['trade_date'] = hs300_data['trade_date'].astype(int)
    
    # 4. 筛选交易数据
    mask = (trade_data["trade_date"] >= start_date_int) & (trade_data["trade_date"] <= end_date_int)
    trade_interval = trade_data[mask].copy()
    
    if trade_interval.empty:
        st.warning("在选定区间内没有交易数据")
        return
    
    # 5. 计算上证指数（上交所股票等权重平均）
    sh_trades = trade_interval[trade_interval['ts_code'].isin(sh_stocks)]
    
    if not sh_trades.empty:
        # 按日期和股票分组，计算每个股票每个交易日的收盘价
        sh_daily = sh_trades.pivot_table(
            index='trade_date', 
            values='close', 
            aggfunc='mean'
        ).reset_index().sort_values('trade_date')
        
        sh_daily.columns = ['trade_date', '上证A股指数']
    else:
        st.warning("没有上交所股票交易数据")
        sh_daily = pd.DataFrame(columns=['trade_date', '上证A股指数'])
    
    # 6. 计算深证指数（深交所股票等权重平均）
    sz_trades = trade_interval[trade_interval['ts_code'].isin(sz_stocks)]
    
    if not sz_trades.empty:
        sz_daily = sz_trades.pivot_table(
            index='trade_date', 
            values='close', 
            aggfunc='mean'
        ).reset_index().sort_values('trade_date')
        
        sz_daily.columns = ['trade_date', '深证A股指数']
    else:
        st.warning("没有深交所股票交易数据")
        sz_daily = pd.DataFrame(columns=['trade_date', '深证A股指数'])
    
    # 7. 处理沪深300数据
    hs300_daily = hs300_data[
        (hs300_data['trade_date'] >= start_date_int) & 
        (hs300_data['trade_date'] <= end_date_int)
    ].sort_values('trade_date')[['trade_date', 'close']]
    hs300_daily.columns = ['trade_date', '沪深300指数']
    
    # 8. 合并三个指数的数据
    # 找到所有交易日的并集
    all_dates = set()
    for df in [sh_daily, sz_daily, hs300_daily]:
        if not df.empty:
            all_dates.update(df['trade_date'].tolist())
    
    all_dates = sorted(list(all_dates))
    
    # 创建合并的数据框
    merged_data = pd.DataFrame({'trade_date': all_dates})
    
    # 合并各指数数据
    merged_data = merged_data.merge(sh_daily, on='trade_date', how='left')
    merged_data = merged_data.merge(sz_daily, on='trade_date', how='left')
    merged_data = merged_data.merge(hs300_daily, on='trade_date', how='left')
    
    # 向前填充缺失值
    merged_data = merged_data.fillna(method='ffill').fillna(method='bfill')
    
    if merged_data.empty:
        st.warning("没有足够的数据绘制图表")
        return
    
    # 9. 创建图表
    dates = pd.to_datetime(merged_data['trade_date'].astype(str), format='%Y%m%d')
    
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=(
            f'上证A股指数 ({len(sh_stocks)}只股票)',
            f'深证A股指数 ({len(sz_stocks)}只股票)', 
            '沪深300指数'
        ),
        horizontal_spacing=0.1
    )
    
    # 图1：上证指数
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=merged_data['上证A股指数'],
            mode='lines',
            name='上证指数',
            line=dict(color='#FF6B6B', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>指数: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 图2：深证指数
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=merged_data['深证A股指数'],
            mode='lines',
            name='深证指数',
            line=dict(color='#4ECDC4', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>指数: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 图3：沪深300指数
    fig.add_trace(
        go.Scatter(
            x=dates, 
            y=merged_data['沪深300指数'],
            mode='lines',
            name='沪深300',
            line=dict(color='#45B7D1', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>指数: %{y:.2f}<extra></extra>'
        ),
        row=1, col=3
    )
    
    # 10. 更新图表布局
    fig.update_layout(
        height=450,
        showlegend=False,
        title_text=f"主要股票价格指数走势图 ({dates.iloc[0].strftime('%Y-%m-%d')} 至 {dates.iloc[-1].strftime('%Y-%m-%d')})",
        title_font=dict(size=16, color='#2E4057'),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor='white'
    )
    
    # 设置坐标轴
    for i in range(1, 4):
        fig.update_xaxes(
            title_text="日期",
            tickformat="%m-%d",
            tickangle=45,
            row=1, col=i,
            gridcolor='rgba(128, 128, 128, 0.1)',
            showgrid=True
        )
        fig.update_yaxes(
            title_text="指数值",
            row=1, col=i,
            gridcolor='rgba(128, 128, 128, 0.1)',
            showgrid=True
        )
    
    # 显示图表
    st.plotly_chart(fig, use_container_width=True)
    
    # 11. 显示统计信息
    st.markdown("#### 📊 指数统计信息")
    
    # 显示详细数据
    with st.expander("📈 点击查看详细数据"):
        tab1, tab2, tab3 = st.tabs(["📊 指数数据", "📈 相关性分析", "📋 股票列表"])
        
        with tab1:
            st.markdown("##### 合并指数数据")
            display_df = merged_data.copy()
            display_df['trade_date'] = pd.to_datetime(display_df['trade_date'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
            st.dataframe(display_df,use_container_width=True)
        
        with tab2:
            st.markdown("##### 指数相关性分析")
            correlation = merged_data[['上证A股指数', '深证A股指数', '沪深300指数']].corr()
            st.dataframe(correlation.style.background_gradient(cmap='coolwarm', axis=None))
            
            # 创建相关性热图
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=correlation.values.round(3),
                texttemplate='%{text}',
                textfont={"size": 12}
            ))
            
            fig_corr.update_layout(
                title="指数相关性热图",
                height=400,
                xaxis_title="指数",
                yaxis_title="指数"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 上交所股票列表")
                st.dataframe(pd.DataFrame({'股票代码': sh_stocks[:50]}))
                st.write(f"共 {len(sh_stocks)} 只上交所股票")
            
            with col2:
                st.markdown("##### 深交所股票列表")
                st.dataframe(pd.DataFrame({'股票代码': sz_stocks[:50]}))
                st.write(f"共 {len(sz_stocks)} 只深交所股票")

# ================== 技术指标计算函数 ==================
def calculate_technical_indicators(df):
    """计算技术指标"""
    df = df.sort_values('trade_date').copy()
    
    # 1. MA移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 2. MACD
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['DIF'] = df['EMA12'] - df['EMA26']
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = (df['DIF'] - df['DEA']) * 2
    
    # 3. KDJ
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
    df['K'] = df['RSV'].ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # 4. RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. OBV
    df['OBV'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
    
    # 6. 涨跌趋势指标 (1表示上涨，0表示下跌)
    df['trend'] = (df['close'].diff() > 0).astype(int)
    
    return df

def prepare_training_data(tech_data):
    """准备训练数据"""
    df = tech_data.dropna().copy()
    
    # 特征列
    feature_cols = ['MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 'K', 'D', 'J', 'RSI', 'OBV']
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    if not feature_cols:
        return None, None, []
    
    # 目标列（预测下一日的涨跌）
    df['target'] = df['trend'].shift(-1)
    df = df.dropna()
    
    if df.empty:
        return None, None, []
    
    # 特征和目标
    X = df[feature_cols]
    y = df['target']
    
    return X, y, feature_cols

def train_prediction_model(X, y, model_type='逻辑回归'):
    """训练预测模型"""
    if X is None or y is None or len(X) == 0:
        return None
    
    try:
        # 划分数据集：70%训练集，20%测试集，10%预测集
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_pred, y_test, y_pred = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)  # 1/3 of 30% = 10%
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_pred_scaled = scaler.transform(X_pred)
        
        # 选择模型
        if model_type == '逻辑回归':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == '随机森林':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == '支持向量机':
            model = SVC(random_state=42, probability=True)
        elif model_type == '神经网络':
            model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        else:  # 梯度提升树
            model = GradientBoostingClassifier(random_state=42)
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # 计算准确率
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        return {
            'model': model,
            'scaler': scaler,
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'X_pred': X_pred, 'y_pred_actual': y_pred,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }
    except Exception as e:
        st.error(f"模型训练失败: {str(e)}")
        return None

# ================== 综合评价函数 ==================
def Fr(data, year):
    """综合评价函数"""
    if data.empty:
        return pd.DataFrame(columns=['股票代码', '股票简称', '综合得分'])
    
    tdata = data[data['年度'] == year]
    if tdata.empty:
        return pd.DataFrame(columns=['股票代码', '股票简称', '综合得分'])

    # 1.空值和负值处理
    data_x = tdata.iloc[:, 1:-1]
    data_x = data_x[data_x > 0]  # 过滤负值
    data_x['股票代码'] = tdata['股票代码'].values
    data_x = data_x.dropna()  # 删除空值

    if data_x.empty:
        return pd.DataFrame(columns=['股票代码', '股票简称', '综合得分'])

    # 2.标准化
    X = data_x.iloc[:, :-1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3.主成分分析
    pca = PCA(n_components=0.95)
    Y = pca.fit_transform(X_scaled)
    gxl = pca.explained_variance_ratio_

    # 4.综合得分
    F = (Y * gxl).sum(axis=1)

    # 5.获取股票简称（重点修改：增强容错）
    result_df = pd.DataFrame({
        '股票代码': data_x['股票代码'].values,
        '综合得分': F
    })
    
    # 6.添加股票简称（重构这部分逻辑）
    try:
        # 读取股票基本信息
        stock_info = pd.read_excel('上市公司基本信息.xlsx')
        # 统一列名格式
        stock_info.columns = stock_info.columns.str.strip()
        
        # 处理不同的列名情况
        if 'ts_code' in stock_info.columns and 'name' in stock_info.columns:
            stock_info_renamed = stock_info.rename(columns={'ts_code': '股票代码', 'name': '股票简称'})
        elif '股票代码' in stock_info.columns and '股票简称' in stock_info.columns:
            stock_info_renamed = stock_info
        else:
            # 如果列名不匹配，用股票代码作为简称
            result_df['股票简称'] = result_df['股票代码'].astype(str)
            stock_info_renamed = None
        
        # 合并股票简称
        if stock_info_renamed is not None:
            # 确保股票代码格式一致（去除空格/大小写）
            result_df['股票代码'] = result_df['股票代码'].astype(str).str.strip()
            stock_info_renamed['股票代码'] = stock_info_renamed['股票代码'].astype(str).str.strip()
            
            result_df = pd.merge(
                result_df, 
                stock_info_renamed[['股票代码', '股票简称']], 
                on='股票代码', 
                how='left'
            )
        
        # 填充缺失的简称（用股票代码）
        result_df['股票简称'] = result_df['股票简称'].fillna(result_df['股票代码'].astype(str))
        
    except Exception as e:
        # 读取失败时，直接用股票代码作为简称
        result_df['股票简称'] = result_df['股票代码'].astype(str)
        print(f"读取股票简称失败: {e}")
    
    result_df = result_df.sort_values('综合得分', ascending=False).reset_index(drop=True)
    return result_df

# ================== 收益率计算函数 ==================
def Tr(rdata, rank, date1, date2, trade_data):
    """
    收益率计算函数
    参数：
        rdata: 综合评价结果DataFrame（含股票代码、股票简称、综合得分）
        rank: 要计算的股票数量
        date1: 开始日期（date对象）
        date2: 结束日期（date对象）
        trade_data: 合并后的交易数据DataFrame
    返回：
        stock_ret_df: 个股收益率DataFrame
        avg_return: 投资组合平均收益率
        hs300_return: 同期沪深300收益率
        valid_count: 有效计算收益率的股票数量
    """
    # 初始化变量 =====================
    stock_ret_list = []  # 存储个股收益率结果
    total_return = 0.0   # 所有有效股票收益率总和
    valid_count = 0      # 有效计算收益率的股票数量
    hs300_return = 0.0   # 沪深300收益率
    
    # 数据预处理 =====================
    # 复制交易数据避免修改原数据
    A = trade_data.copy()
    
    # 统一日期格式：转为字符串（YYYYMMDD），去除空格
    A['trade_date'] = A['trade_date'].astype(str).str.strip()
    date1_str = date1.strftime("%Y%m%d")
    date2_str = date2.strftime("%Y%m%d")
    
    # 统一股票代码格式：大写、去空格
    A['ts_code'] = A['ts_code'].astype(str).str.strip().str.upper()
    
    # 获取排名前N的股票代码
    top_stk = rdata.head(rank)['股票代码'].tolist()
    
    # 调试信息=====================
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### 📋 待计算股票列表")
        st.dataframe(rdata.head(rank)[['股票代码', '股票简称']], use_container_width=True)
    with col2:
        st.write("#### 📅 时间区间")
        st.write(f"- 开始日期：{date1_str}")
        st.write(f"- 结束日期：{date2_str}")
        st.write(f"- trade_data时间范围：{A['trade_date'].min()} ~ {A['trade_date'].max()}")
        # 检查时间区间是否重叠
        if date1_str > A['trade_date'].max() or date2_str < A['trade_date'].min():
            st.warning("⚠️ 选定时间区间完全超出交易数据范围！")
    
    # 遍历计算个股收益率 =====================
    for code in top_stk:
        # 初始化个股信息
        stock_name = ""
        ret_info = {
            '股票代码': code,
            '股票简称': "",
            '收益率(%)': "数据不足"
        }
        
        # 1. 获取股票简称
        stock_info = rdata[rdata['股票代码'] == code]
        if not stock_info.empty:
            stock_name = stock_info['股票简称'].values[0]
        else:
            stock_name = str(code).split('.')[0]  # 无简称时用代码前缀
        
        ret_info['股票简称'] = stock_name
        
        # 2. 生成所有可能的代码匹配格式（解决格式不匹配问题）
        base_code = str(code).strip().upper()
        search_codes = []
        
        # 情况1：原代码带后缀（如600000.SH）→ 匹配带后缀+纯数字
        if '.' in base_code:
            search_codes.append(base_code)                  # 600000.SH
            search_codes.append(base_code.split('.')[0])    # 600000
        # 情况2：原代码纯数字（如600000）→ 匹配纯数字+常见后缀
        else:
            search_codes.append(base_code)                  # 600000
            search_codes.extend([f"{base_code}.SH", f"{base_code}.SZ", f"{base_code}.BJ"])
        
        # 去重+统一大写
        search_codes = list(set([s.upper() for s in search_codes]))
        
        # 3. 查找该股票在时间区间内的交易数据
        stk_data = None
        for search_code in search_codes:
            # 模糊匹配：包含核心代码（避免后缀/格式问题）
            temp_data = A[
                (A['ts_code'].str.contains(search_code.split('.')[0], na=False)) & 
                (A['trade_date'] >= date1_str) & 
                (A['trade_date'] <= date2_str)
            ].sort_values('trade_date')
            
            if not temp_data.empty:
                stk_data = temp_data
                break
        
        # 4. 检查交易数据是否有效
        if stk_data is None or stk_data.empty:
            #ret_info['备注'] += f"未找到匹配的交易数据（尝试匹配：{search_codes}）；"
            stock_ret_list.append(ret_info)
            continue
        
        # 5. 检查数据行数（至少1行）
        if len(stk_data) < 1:
            stock_ret_list.append(ret_info)
            continue
        
        # 6. 提取收盘价并计算收益率
        p1 = stk_data['close'].iloc[0]  # 区间首日收盘价
        p2 = stk_data['close'].iloc[-1] # 区间末日收盘价
        
        # 检查价格是否有效（非空、非0）
        if pd.isna(p1) or pd.isna(p2) or p1 == 0:
            #ret_info['备注'] += f"价格数据无效（首日：{p1}，末日：{p2}）；"
            stock_ret_list.append(ret_info)
            continue
        
        # 计算收益率
        ri = (p2 - p1) / p1
        ret_info['收益率(%)'] = f"{ri * 100:.2f}%"
        total_return += ri
        valid_count += 1
        
        # 7. 添加到结果列表
        stock_ret_list.append(ret_info)
    
    # 计算投资组合平均收益率 =====================
    avg_return = total_return / valid_count if valid_count > 0 else 0.0
    
    # 计算沪深300收益率 =====================
    hs300 = pd.read_excel('沪深300指数交易数据.xlsx')
    hs300['trade_date'] = hs300['trade_date'].astype(str).str.strip()
    
    # 筛选时间区间内的沪深300数据
    hs300_data = hs300[
        (hs300['trade_date'] >= date1_str) & 
        (hs300['trade_date'] <= date2_str)
    ].sort_values('trade_date')
    
    if len(hs300_data) >= 2:
        start_price = hs300_data['close'].iloc[0]
        end_price = hs300_data['close'].iloc[-1]
        if start_price > 0 and not pd.isna(start_price) and not pd.isna(end_price):
            hs300_return = (end_price - start_price) / start_price
    else:
        st.warning("⚠️ 沪深300数据行数不足，收益率重置为0")
    
    # 生成结果DataFrame =====================
    stock_ret_df = pd.DataFrame(stock_ret_list)
    
    return stock_ret_df, avg_return, hs300_return, valid_count

# 页面展示函数
def st_fig():
    # 读取行业分类数据
    try:
        info = pd.read_excel('最新个股申万行业分类(完整版-截至7月末).xlsx')
        nm_L1 = list(set(info['新版一级行业'].values))
    except FileNotFoundError:
        st.warning("未找到行业分类文件，仅显示市场总览")
        nm_L1 = []
    
    nm_L = ['市场总览'] + nm_L1
    
    # 页面配置
    st.set_page_config(
        page_title="金融数据挖掘及其应用综合实训",
        layout='wide',
        initial_sidebar_state="expanded"
    )
    
    # 自定义CSS样式 - 紫色背景
    st.markdown("""
        <style>
        .main { padding-top: 1rem; }
        h1, h2, h3 { color: #2E4057; font-family: "Microsoft YaHei", sans-serif; }
        .sidebar .sidebar-content { background-color: #F8F9FA; padding-top: 1rem; }
                                   
        /* 添加淡紫色背景 */
        .stApp {
            background-color: #E6E6FA;
        }
        </style>
        """, unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.markdown("<h3 style='color: #2E4057;'>📋 数据分类选择</h3>", unsafe_allow_html=True)
        nm = st.selectbox(
            label="请选择查看的分类",
            options=nm_L,
            index=0,
            label_visibility="collapsed",
            placeholder="选择分类..."
        )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <div style='font-size: 0.9rem; color: #6C757D;'>
            <p>👤 姓名：iuu_star</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 市场总览模块
    if nm == '市场总览':
        st.markdown(f"<h2 style='color: #2E4057;'>📊 {nm}</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-bottom: 1rem;'>", unsafe_allow_html=True)
        
        t1, t2 = st.tabs(["📈 主要市场指数行情", "📊 行业统计分析"])
        
        with t1:
            st.markdown("#### 📅 时间区间选择")
            min_date = date(2022, 1, 1)
            max_date = date(2025, 12, 11)
            col1, col2 = st.columns([1, 1], gap="medium")
            
            with col1:
                selected_start_date = st.date_input(
                    "开始日期",
                    value=date(2024, 12, 1),
                    min_value=min_date,
                    max_value=max_date,
                    key='start_date1',
                    help="选择统计开始日期"
                )
            
            with col2:    
                default_end_date = selected_start_date + timedelta(days=30)   #默认结束日期为开始日期后1个月
                default_end_date = max(min(default_end_date, max_date), selected_start_date)
                
                selected_end_date = st.date_input(
                    "结束日期",
                    value=default_end_date,
                    min_value=selected_start_date,
                    max_value=max_date,
                    key='end_date1',
                    help="选择统计结束日期"
                )
                
            # 检查时间跨度
            days_diff = (selected_end_date - selected_start_date).days
            if days_diff > 90:
                st.warning(f"时间跨度{days_diff}天可能过长，龙虎榜通常统计短期异常波动")
                st.info("建议选择时间跨度在30天以内")
                
            # 将date对象转换为整数格式
            start_date_int = date_to_int(selected_start_date)
            end_date_int = date_to_int(selected_end_date)
            
            # 指数走势图模块
            st.markdown("#### 📈 主要股票价格指数走势图")
            st.markdown("**上证A股指数、深证A股指数、沪深300指数**（1×3子图展示）")
            
            # 绘制指数走势图
            plot_stock_index_charts_actual(start_date_int, end_date_int, trade_data)
            
            # 龙虎榜统计模块
            st.markdown("#### 📉 龙虎榜统计分析")
            
            # 检查数据是否加载成功
            if trade_data.empty:
                st.error("交易数据加载失败，无法计算龙虎榜")
            else:
                # 将date对象转换为整数格式
                start_date_int = date_to_int(selected_start_date)
                end_date_int = date_to_int(selected_end_date)
                                
                # 调用函数获取龙虎榜数据，传递trade_data参数
                with st.spinner("正在计算龙虎榜数据..."):
                    up_20, down_20 = calculate_dragon_tiger(start_date_int, end_date_int, trade_data)
                
                col1, col2 = st.columns(2, gap="medium")
                
                with col1:
                    st.subheader('📈 累计涨幅大于20%的股票')
                    if up_20.empty:
                        st.info("统计区间内无累计涨幅大于20%的股票")
                    else:
                        st.markdown(f"**共找到 {len(up_20)} 只涨幅大于20%的股票**")
                        st.dataframe(up_20, use_container_width=True)
                
                with col2:
                    st.subheader('📉 累计跌幅大于20%的股票')
                    if down_20.empty:
                        st.info("统计区间内无累计跌幅大于20%的股票")
                    else:
                        st.markdown(f"**共找到 {len(down_20)} 只跌幅大于20%的股票**")
                        st.dataframe(down_20, use_container_width=True)
        
        with t2:
            # 实训3内容：行业财务统计
            if SHIXUN3_AVAILABLE:
                display_financial_analysis()
            else:
                st.warning("实训3模块未找到，请确保shixun3.py文件存在")
    
    # 其他行业模块（所有行业都使用相同的结构）
    else:  # 如果不是市场总览，就是具体的行业
        st.markdown(f"<h2 style='color: #2E4057;'>{nm}行业分析</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin-bottom: 1rem;'>", unsafe_allow_html=True)
        
        # 顶部图表区域
        left, right = st.columns(2, gap="medium")
        
        with left:
            st.subheader('📈 行业指数走势图')
            data = pd.read_csv('index_trdata.csv')
            data_i = data[data['name'] == nm].sort_values('trade_date')
            
            if not data_i.empty:
                dates = pd.to_datetime(data_i['trade_date'].astype(str), format='%Y%m%d')
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=dates,
                    y=data_i['close'],
                    mode='lines',
                    name=f'{nm}指数',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='日期: %{x|%Y-%m-%d}<br>指数: %{y:.2f}<extra></extra>'
                ))
                
                fig1.update_layout(
                    title=f'申万{nm}行业指数走势图',
                    xaxis_title='日期',
                    yaxis_title='收盘指数',
                    height=300,
                    hovermode='x unified',
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info(f"暂无{nm}行业指数数据")
        
        with right:
            st.subheader('📊 前6只股票价格走势图')
            info = pd.read_excel('最新个股申万行业分类(完整版-截至7月末).xlsx')
            trdata = pd.read_csv('stk_trdata.csv')
            
            industry_info = info[info['新版一级行业'] == nm]
            if not industry_info.empty:
                industry_codes = industry_info.iloc[:6, 2]  # 第3列为股票代码
                industry_names = industry_info.iloc[:6, 3]  # 第4列为股票简称
                
                fig2 = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=[f"{name}" for name in industry_names[:6]],
                    vertical_spacing=0.15,
                    horizontal_spacing=0.1
                )
                
                for idx, (code, name) in enumerate(zip(industry_codes[:6], industry_names[:6])):
                    row = idx // 2 + 1
                    col = idx % 2 + 1
                    
                    stock_data = trdata[trdata['ts_code'] == code].sort_values('trade_date')
                    
                    if not stock_data.empty:
                        dates = pd.to_datetime(stock_data['trade_date'].astype(str), format='%Y%m%d')
                        
                        fig2.add_trace(
                            go.Scatter(
                                x=dates,
                                y=stock_data['close'],
                                mode='lines',
                                name=name,
                                line=dict(width=1.5, color='#ff7f0e'),
                                hovertemplate='日期: %{x|%Y-%m-%d}<br>价格: %{y:.2f}<extra></extra>'
                            ),
                            row=row, col=col
                        )
                
                fig2.update_layout(
                    height=400,
                    showlegend=False,
                    title_text=f"{nm}行业前6只股票价格走势",
                    plot_bgcolor='white'
                )
                
                for i in range(1, 7):
                    row = (i-1)//2 + 1
                    col = (i-1)%2 + 1
                    fig2.update_xaxes(title_text="日期", row=row, col=col)
                    fig2.update_yaxes(title_text="价格", row=row, col=col)
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info(f"暂无{nm}行业股票数据")
        
        # 行业数据标签页
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 行业指数交易数据", 
            "🏢 行业上市公司信息", 
            "📊 行业股票交易数据",
            "💰 行业股票财务数据"
        ])
        
        with tab1:
            st.markdown(f"#### {nm}行业指数交易数据详情")
            try:
                data = pd.read_csv('index_trdata.csv')
                industry_data = data[data['name'] == nm].sort_values('trade_date')
                if not industry_data.empty:
                    industry_data.columns = ['股票代码', '行业简称', '交易日期', '开盘指数', '收盘指数', '成交量', '市盈率', '市净率']
                    st.dataframe(industry_data, use_container_width=True)
                else:
                    st.info(f"暂无{nm}行业指数交易数据")
            except:
                st.info("数据加载中...")
        
        with tab2:
            st.markdown(f"#### {nm}行业上市公司基本信息")
            try:
                co_data = pd.read_excel('上市公司基本信息.xlsx')
                info = pd.read_excel('最新个股申万行业分类(完整版-截至7月末).xlsx')
                
                industry_info = info[info['新版一级行业'] == nm]
                if not industry_info.empty:
                    industry_co_data = pd.merge(co_data, industry_info[['股票代码']], 
                                               left_on='ts_code', right_on='股票代码', how='inner')
                    st.dataframe(industry_co_data, use_container_width=True)
                    st.write(f"共 {len(industry_co_data)} 家{nm}行业上市公司")
                else:
                    st.info(f"暂无{nm}行业上市公司信息")
            except:
                st.info("数据加载中...")
        
        with tab3:
            st.markdown(f"#### {nm}行业股票交易数据")
            try:
                trdata = pd.read_csv('stk_trdata.csv')
                info = pd.read_excel('最新个股申万行业分类(完整版-截至7月末).xlsx')
                
                industry_info = info[info['新版一级行业'] == nm]
                if not industry_info.empty:
                    industry_trdata = pd.merge(trdata, industry_info[['股票代码', '股票简称']], 
                                              on='股票代码', how='inner')
                    industry_trdata = industry_trdata.sort_values(['股票代码', 'trade_date'])
                    st.dataframe(industry_trdata.head(100), use_container_width=True)
                    st.write(f"共 {len(industry_trdata)} 条交易记录，{len(industry_trdata['股票代码'].unique())} 只股票")
                else:
                    st.info(f"暂无{nm}行业股票交易数据")
            except:
                st.info("数据加载中...")
        
        with tab4:
            st.markdown(f"#### {nm}行业股票财务数据")
            try:
                findata = pd.read_csv('fin_data.csv')
                info = pd.read_excel('最新个股申万行业分类(完整版-截至7月末).xlsx')
                
                industry_info = info[info['新版一级行业'] == nm]
                if not industry_info.empty:
                    industry_findata = pd.merge(findata, industry_info[['股票代码']], 
                                               on='股票代码', how='inner')
                    st.dataframe(industry_findata, use_container_width=True)
                    st.write(f"共 {len(industry_findata)} 条财务记录，{len(industry_findata['股票代码'].unique())} 只股票")
                else:
                    st.info(f"暂无{nm}行业财务数据")
            except:
                st.info("数据加载中...")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # 分析标签页
        tb1, tb2 = st.tabs(["📝 综合评价分析", "🔍 股票价格涨跌趋势分析"])
        
        # ========== tb1: 综合评价分析 ==========
        with tb1:
            st.subheader('🎯 综合评价分析')
            
            # 参数设置
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                year = st.selectbox("**📅 评价年度**", [2022, 2023, 2024], 
                                   help="选择财务数据年度", key=f'year_select_{nm}')
            
            with col2:
                rank = st.selectbox("**🏆 排名数量**", [5, 10, 15, 20], 
                                   help="选择展示的股票数量", key=f'rank_select_{nm}')
            
            with col3:
                st.markdown("**📅 开始日期**")
                min_date = date(2022, 1, 1)
                max_date = date(2025, 12, 11)
                selected_start_date = st.date_input(
                    "",
                    value=date(2024, 1, 1),
                    min_value=min_date,
                    max_value=max_date,
                    key=f'start_date_{nm}',
                    help="选择持有期开始日期",
                    label_visibility="collapsed"
                )
            
            with col4:
                st.markdown("**📅 结束日期**")
                default_end_date = add_six_months(selected_start_date)
                default_end_date = max(min(default_end_date, max_date), selected_start_date)
                selected_end_date = st.date_input(
                    "",
                    value=default_end_date,
                    min_value=selected_start_date,
                    max_value=max_date,
                    key=f'end_date_{nm}',
                    help="选择持有期结束日期",
                    label_visibility="collapsed"
                )
            
            # 综合排名结果显示区域
            st.markdown("#### 📊 综合排名结果")
            
            try:
                # 读取财务数据
                findata = pd.read_csv('fin_data.csv')
                
                # 筛选当前行业数据
                info = pd.read_excel('最新个股申万行业分类(完整版-截至7月末).xlsx')
                industry_stocks = info[info['新版一级行业'] == nm][['股票代码']]
                
                # 合并财务数据
                findata_industry = pd.merge(findata, industry_stocks, how='inner', on='股票代码')
                
                # 计算综合评价
                eval_result = Fr(findata_industry, year)
                
                if not eval_result.empty and len(eval_result) > 0:
                    # ========== 1. 并排显示综合排名和得分分布 ==========
                    col_left, col_right = st.columns([1, 1], gap="large")
                    
                    with col_left:
                        st.markdown("##### 🏆 综合排名结果")
                        display_df = eval_result.head(rank).copy()
                        display_df['排名'] = range(1, len(display_df) + 1)
                        display_df['综合得分'] = display_df['综合得分'].apply(lambda x: f"{x:.4f}")
                        display_df = display_df[['排名', '股票代码', '股票简称', '综合得分']]
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=min(400, 50 + len(display_df) * 35),
                            hide_index=True
                        )
                    
                    with col_right:
                        st.markdown(f"##### 📈 前{rank}只股票综合得分分布")
                        import matplotlib.pyplot as plt
                        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
                        plt.rcParams['axes.unicode_minus'] = False
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        top_data = eval_result.head(rank)
                        
                        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_data)))
                        bars = ax.bar(range(len(top_data)), top_data['综合得分'], color=colors, alpha=0.8)
                        
                        for i, (idx, row) in enumerate(top_data.iterrows()):
                            ax.text(i, row['综合得分'] + 0.02, f"{row['综合得分']:.2f}", 
                                   ha='center', fontsize=9, fontweight='bold')
                        
                        ax.set_title(f"{nm}行业前{rank}只股票综合得分", fontsize=14, fontweight='bold', pad=20)
                        ax.set_xlabel("股票简称", fontsize=11)
                        ax.set_ylabel("综合得分", fontsize=11)
                        ax.set_xticks(range(len(top_data)))
                        ax.set_xticklabels(top_data['股票简称'].tolist(), rotation=45, ha='right', fontsize=10)
                        ax.grid(axis='y', alpha=0.3, linestyle='--')
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # ========== 2. 收益率分析 ==========
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("### 📈 收益率分析")
                    
                    # 计算收益率
                    stock_ret_df, portfolio_return, hs300_return, valid_count = Tr(eval_result, rank, selected_start_date, selected_end_date, trade_data)
                    
                    if not stock_ret_df.empty:
                        # 2.1 投资组合的个股收益率
                        st.markdown("#### 📊 投资组合个股收益率")
                        
                        st.dataframe(
                            stock_ret_df,
                            use_container_width=True,
                            height=min(400, 60 + len(stock_ret_df) * 35)
                        )
                        
                        # 2.2 收益率对比展示
                        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            portfolio_color = '#28a745' if portfolio_return >= 0 else '#dc3545'
                            st.markdown(f"""
                            <h4 style='margin: 0; color: #2E4057;'>投资组合总收益率</h4>
                            <h2 style='margin: 10px 0; color: {portfolio_color};'>{portfolio_return:+.2%}</h2>

                            """, unsafe_allow_html=True)
                        
                        with col2:
                            hs300_color = '#28a745' if hs300_return >= 0 else '#dc3545'
                            st.markdown(f"""
                            <h4 style='margin: 0; color: #2E4057;'>同期沪深300收益率</h4>
                            <h2 style='margin: 10px 0; color: {hs300_color};'>{hs300_return:+.2%}</h2>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            alpha = portfolio_return - hs300_return
                            alpha_color = '#28a745' if alpha >= 0 else '#dc3545'
                            st.markdown(f"""
                            <h4 style='margin: 0; color: #2E4057;'>超额收益 (Alpha)</h4>
                            <h2 style='margin: 10px 0; color: {alpha_color};'>{alpha:+.2%}</h2>
                            """, unsafe_allow_html=True)
                        
                        # 2.3 收益率对比图表
                        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                        st.markdown("#### 📈 收益率对比分析")
                    
                        
                        # 提取有效的收益率数据
                        valid_returns = []
                        stock_names = []
                        for _, row in stock_ret_df.iterrows():
                            ret_str = str(row['收益率(%)']).strip()
                            if ret_str in ['数据不足', 'N/A', '', 'nan', 'None'] or pd.isna(ret_str):
                                continue
                            
                            ret_value = float(ret_str.replace('%', '').replace('+', ''))
                            valid_returns.append(ret_value)
                            stock_name = str(row['股票简称']).strip() if pd.notna(row['股票简称']) else f"股票{_}"
                            stock_names.append(stock_name)
                        
                        if valid_returns and len(valid_returns) > 0:
                            # 计算组合平均和沪深300收益率（转为百分比）
                            avg_return_pct = sum(valid_returns) / len(valid_returns)
                            hs300_return_pct = hs300_return * 100
                        
                            # 构造分组数据 =====
                            # 1. 个股收益率
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=stock_names,
                                y=valid_returns,
                                name='个股收益率',
                                marker=dict(
                                    color=['#2ECC71' if r > 0 else '#E74C3C' for r in valid_returns],
                                    line=dict(color='#FFFFFF', width=1),
                                    opacity=0.8
                                ),
                                hovertemplate='%{x}<br>收益率：%{y:.2f}%<extra></extra>'
                            ))
                        
                            # 2. 组合平均收益率（同维度对比）
                            fig.add_trace(go.Bar(
                                x=stock_names,
                                y=[avg_return_pct] * len(stock_names),
                                name='组合平均',
                                marker=dict(
                                    color='#F39C12',
                                    line=dict(color='#FFFFFF', width=1),
                                    opacity=0.6
                                ),
                                hovertemplate='组合平均收益率：%{y:.2f}%<extra></extra>'
                            ))
                        
                            # 3. 沪深300收益率（同维度对比）
                            fig.add_trace(go.Bar(
                                x=stock_names,
                                y=[hs300_return_pct] * len(stock_names),
                                name='沪深300',
                                marker=dict(
                                    color='#3498DB',
                                    line=dict(color='#FFFFFF', width=1),
                                    opacity=0.6
                                ),
                                hovertemplate='沪深300收益率：%{y:.2f}%<extra></extra>'
                            ))
                        
                            # 美化布局 =====
                            fig.update_layout(
                                title=dict(
                                    text=f'{nm}行业股票收益率对比',
                                    font=dict(size=16, weight='bold', color='#2C3E50'),
                                    x=0.5
                                ),
                                xaxis_title=dict(text='股票简称', font=dict(size=12, color='#34495E')),
                                yaxis_title=dict(text='收益率 (%)', font=dict(size=12, color='#34495E')),
                                height=500,
                                barmode='group',  # 分组显示（而非堆叠）
                                bargap=0.1,       # 组内间距
                                bargroupgap=0.1,  # 组间间距
                                plot_bgcolor='#F8F9FA',
                                paper_bgcolor='#FFFFFF',
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=-0.3,
                                    xanchor='center',
                                    x=0.5,
                                    font=dict(size=10)
                                ),
                                hoverlabel=dict(
                                    bgcolor='#FFFFFF',
                                    bordercolor='#DDDDDD',
                                    font=dict(size=10)
                                )
                            )
                        
                            # 坐标轴优化 =====
                            fig.update_xaxes(
                                tickangle=45,
                                tickfont=dict(size=10, color='#7F8C8D'),
                                gridcolor='#EEEEEE'
                            )
                            fig.update_yaxes(
                                tickfont=dict(size=10, color='#7F8C8D'),
                                gridcolor='#EEEEEE',
                                zerolinecolor='#DDDDDD',
                                zerolinewidth=1
                            )
                        
                            # 渲染图表
                            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                        else:
                            st.info("📊 暂无有效收益率数据")
                            
                    
                else:
                    st.info("📝 暂无综合评价数据")
                    
            except Exception as e:
                st.error(f"❌ 综合评价计算出错: {str(e)}")

        # 股票价格涨跌趋势分析 ==========
        with tb2: 
            st.subheader('📉 股票价格涨跌趋势分析')
            
            # 参数选择
            col1, col2 = st.columns([1, 3], gap="medium")
            with col1:
                year1 = st.selectbox("📅 年度", [2022, 2023, 2024], key=f'y2_{nm}')
                rank1 = st.selectbox("🏆 排名数量", [5, 10, 15, 20], key=f'r2_{nm}')
            
            #  3. 综合排名股票交易数据 ==========
            with col2:
                st.markdown("#### 📋 综合排名股票交易数据详情")
                
                try:
                    # 读取财务数据
                    findata = pd.read_csv('fin_data.csv')
                    info = pd.read_excel('最新个股申万行业分类(完整版-截至7月末).xlsx')
                    
                    # 筛选当前行业数据 - 根据实际列名
                    if 'ts_code' in info.columns:
                        code_col = 'ts_code'
                    elif '股票代码' in info.columns:
                        code_col = '股票代码'
                    else:
                        # 查找包含"代码"的列
                        code_col = None
                        for col in info.columns:
                            if '代码' in str(col):
                                code_col = col
                                break
                        if code_col is None:
                            st.error("未找到股票代码列")
                            st.write("行业分类文件列名:", info.columns.tolist())
                            code_col = info.columns[0]  # 使用第一列作为备用
                    
                    # 筛选当前行业数据
                    industry_stocks = info[info['新版一级行业'] == nm][[code_col]]
                    # 重命名为'股票代码'以便合并
                    industry_stocks = industry_stocks.rename(columns={code_col: '股票代码'})
                    
                    # 检查财务数据列名
                    if '股票代码' not in findata.columns:
                        # 尝试重命名
                        for col in findata.columns:
                            if '代码' in str(col):
                                findata = findata.rename(columns={col: '股票代码'})
                                break
                        # 如果还没有，检查是否有ts_code
                        if '股票代码' not in findata.columns and 'ts_code' in findata.columns:
                            findata = findata.rename(columns={'ts_code': '股票代码'})
                    
                    # 合并财务数据
                    findata_industry = pd.merge(findata, industry_stocks, how='inner', on='股票代码')

                    # 计算综合评价
                    if not findata_industry.empty:
                        eval_result_tb2 = Fr(findata_industry, year1)
                        
                        if not eval_result_tb2.empty and len(eval_result_tb2) > 0:                            
                            # 获取排名靠前的股票代码
                            top_stocks = eval_result_tb2.head(rank1)['股票代码'].tolist()
                            # 读取交易数据
                            try:
                                trdata = pd.read_csv('stk_trdata.csv')
                                
                                # 找到交易数据中的股票代码列
                                if 'ts_code' in trdata.columns:
                                    tr_code_col = 'ts_code'
                                elif '股票代码' in trdata.columns:
                                    tr_code_col = '股票代码'
                                else:
                                    # 查找包含"代码"的列
                                    tr_code_col = None
                                    for col in trdata.columns:
                                        if '代码' in str(col):
                                            tr_code_col = col
                                            break
                                    if tr_code_col is None:
                                        # 如果还找不到，尝试查找股票简称列
                                        for col in trdata.columns:
                                            if '名称' in str(col) or 'name' in str(col).lower():
                                                tr_code_col = col
                                                break
                                
                                if tr_code_col:
                                    # 过滤出排名靠前股票的交易数据
                                    top_stock_data = trdata[trdata[tr_code_col].isin(top_stocks)].copy()
                                    
                                    if len(top_stock_data) == 0:
                                        # 如果直接匹配失败，尝试模糊匹配（不带后缀的代码）
                                        st.write("尝试模糊匹配股票代码...")
                                        # 提取基础代码（去掉.SH等后缀）
                                        base_top_stocks = [str(code).split('.')[0] if '.' in str(code) else str(code) for code in top_stocks]
                                        
                                        # 提取交易数据中的基础代码
                                        trdata_base_codes = []
                                        for code in trdata[tr_code_col]:
                                            if isinstance(code, str):
                                                trdata_base_codes.append(code.split('.')[0] if '.' in code else code)
                                            else:
                                                trdata_base_codes.append(str(code))
                                        
                                        # 创建临时列进行匹配
                                        trdata_temp = trdata.copy()
                                        trdata_temp['base_code'] = trdata_base_codes
                                        top_stock_data = trdata_temp[trdata_temp['base_code'].isin(base_top_stocks)].copy()
                                    
                                    if not top_stock_data.empty:
                                        # 添加股票简称
                                        stock_names = eval_result_tb2.set_index('股票代码')['股票简称'].to_dict()
                                        top_stock_data['股票简称'] = top_stock_data[tr_code_col].map(stock_names)
                                        
                                        # 显示最新的交易数据
                                        # 检查是否有排序用的日期列
                                        date_col = 'trade_date'
                                        if 'trade_date' not in top_stock_data.columns:
                                            for col in top_stock_data.columns:
                                                if 'date' in str(col).lower() or '日期' in str(col):
                                                    date_col = col
                                                    break
                                        
                                        top_stock_data = top_stock_data.sort_values([tr_code_col, date_col], 
                                                                                    ascending=[True, False])
                                        
                                        st.write(f"显示前 {rank1} 只股票的最近交易数据")
                                        
                                        # 显示交易数据
                                        display_cols = [tr_code_col, '股票简称', date_col]
                                        # 添加可能的数值列
                                        possible_value_cols = ['open', 'high', 'low', 'close', 'vol', 
                                                              '开盘', '最高', '最低', '收盘', '成交量',
                                                              'pct_chg', 'amount', 'change']
                                        for col in possible_value_cols:
                                            if col in top_stock_data.columns:
                                                display_cols.append(col)
                                        
                                        available_cols = [col for col in display_cols if col in top_stock_data.columns]
                                        
                                        if available_cols:
                                            # 显示前50条记录
                                            st.dataframe(
                                                top_stock_data[available_cols].head(50),
                                                use_container_width=True,
                                                height=400
                                            )
                                            
                                            # 显示统计信息

                                            if date_col in top_stock_data.columns:
                                                st.write(f"- 日期范围: {top_stock_data[date_col].min()} 至 {top_stock_data[date_col].max()}")
                                        else:
                                            st.warning("交易数据表中缺少必要的列")
                                            st.write("可用的列:", top_stock_data.columns.tolist())
                                    else:
                                        st.info("❌ 没有找到这些股票的交易数据")
                                        
                                        # 显示对比信息
                                        st.write("排名股票代码（前5个）:", top_stocks[:5])
                                        st.write("交易数据中的股票代码示例（前5个）:", 
                                                trdata[tr_code_col].unique()[:5] if tr_code_col in trdata.columns else "无股票代码列")
                                else:
                                    st.error("未找到交易数据中的股票代码列")
                                    st.write("交易数据列名:", trdata.columns.tolist())
                                    
                            except FileNotFoundError:
                                st.error("❌ 未找到交易数据文件 'stk_trdata.csv'")
                            except Exception as e:
                                st.error(f"❌ 读取交易数据失败: {str(e)}")
                                import traceback
                                st.write(traceback.format_exc())
                                
                        else:
                            st.info("📝 暂无综合评价数据")
                    else:
                        st.info("📝 该行业暂无财务数据")
                        
                except Exception as e:
                    st.error(f"❌ 数据处理出错: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())
                    
            
            # 4. 技术指标计算 ==========
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 🧮 技术指标计算")
            
            # 4.1 计算指标说明
            st.markdown("""
            **计算指标：**
            1. **MA（移动平均线）**: MA5, MA10, MA20
            2. **MACD（指数平滑异同移动平均线）**: DIF, DEA, MACD
            3. **KDJ（随机指标）**: K, D, J
            4. **RSI（相对强弱指标）**: 14日RSI
            5. **OBV（能量潮指标）**: 成交量能量潮
            6. **涨跌趋势指标**: 1表示上涨，0表示下跌
            """)
            
            # 创建更健壮的技术指标计算函数
            def calculate_technical_indicators_robust(df):
                """更健壮的技术指标计算函数"""
                if df.empty or 'close' not in df.columns:
                    return df
                
                df = df.sort_values('trade_date').copy()
                
                # 1. 移动平均线
                df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
                df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
                df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
                
                # 2. MACD
                try:
                    df['EMA12'] = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
                    df['EMA26'] = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
                    df['DIF'] = df['EMA12'] - df['EMA26']
                    df['DEA'] = df['DIF'].ewm(span=9, adjust=False, min_periods=1).mean()
                    df['MACD'] = (df['DIF'] - df['DEA']) * 2
                except:
                    pass  # 如果计算失败，跳过
                
                # 3. KDJ（需要high和low）
                try:
                    if all(col in df.columns for col in ['high', 'low', 'close']):
                        low_min = df['low'].rolling(window=9, min_periods=1).min()
                        high_max = df['high'].rolling(window=9, min_periods=1).max()
                        df['RSV'] = (df['close'] - low_min) / (high_max - low_min + 1e-8) * 100
                        df['K'] = df['RSV'].ewm(com=2, min_periods=1).mean()
                        df['D'] = df['K'].ewm(com=2, min_periods=1).mean()
                        df['J'] = 3 * df['K'] - 2 * df['D']
                except:
                    pass
                
                # 4. RSI
                try:
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                    rs = gain / (loss + 1e-8)
                    df['RSI'] = 100 - (100 / (1 + rs))
                except:
                    pass
                
                # 5. OBV（需要vol）
                try:
                    if 'vol' in df.columns:
                        df['OBV'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
                except:
                    pass
                
                # 6. 涨跌趋势指标
                df['trend'] = (df['close'].diff() > 0).astype(int)
                
                return df
            
            def display_technical_indicators(tech_data, stock_name):
                """显示技术指标数据"""
                if tech_data.empty:
                    st.info("无技术指标数据")
                    return
                
                # 显示最近30天的技术指标（或所有数据如果少于30天）
                display_count = min(30, len(tech_data))
                recent_data = tech_data.tail(display_count).copy()
                
                # 格式化日期显示
                if 'trade_date' in recent_data.columns:
                    if pd.api.types.is_datetime64_any_dtype(recent_data['trade_date']):
                        recent_data['trade_date_display'] = recent_data['trade_date'].dt.strftime('%Y-%m-%d')
                    else:
                        recent_data['trade_date_display'] = recent_data['trade_date'].astype(str)
                
                # 选择要显示的列
                display_cols = ['trade_date_display', 'close']
                
                # 添加可用的技术指标列
                tech_indicator_cols = ['MA5', 'MA10', 'MA20', 'DIF', 'DEA', 'MACD', 
                                      'K', 'D', 'J', 'RSI', 'OBV', 'trend']
                
                for col in tech_indicator_cols:
                    if col in recent_data.columns:
                        display_cols.append(col)
                
                if len(display_cols) > 1:  # 除了日期之外还有其他列
                    # 格式化数值显示
                    display_df = recent_data[display_cols].copy()
                    
                    # 重命名列以便显示
                    column_display_names = {
                        'trade_date_display': '交易日期',
                        'close': '收盘价',
                        'MA5': 'MA5',
                        'MA10': 'MA10',
                        'MA20': 'MA20',
                        'DIF': 'DIF',
                        'DEA': 'DEA',
                        'MACD': 'MACD',
                        'K': 'K',
                        'D': 'D',
                        'J': 'J',
                        'RSI': 'RSI',
                        'OBV': 'OBV',
                        'trend': '趋势'
                    }
                    
                    display_df = display_df.rename(columns=column_display_names)
                    
                    # 对数值列进行格式化
                    numeric_cols = ['收盘价', 'MA5', 'MA10', 'MA20', 'DIF', 'DEA', 
                                   'MACD', 'K', 'D', 'J', 'RSI', 'OBV']
                    
                    for col in numeric_cols:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:.4f}" if pd.notnull(x) and not pd.isna(x) else "N/A"
                            )
                    
                    if '趋势' in display_df.columns:
                        display_df['趋势'] = display_df['趋势'].apply(
                            lambda x: "↑" if x == 1 else "↓" if pd.notnull(x) and not pd.isna(x) else "N/A"
                        )
                    
                    # 显示数据表格
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=min(400, 50 + len(display_df) * 30)
                    )
                    
                    # 显示技术指标图表
                    if 'close' in recent_data.columns and recent_data['close'].notna().any():
                        st.markdown("##### 📈 技术指标图表")
                        
                        # 创建价格图表
                        fig_price = go.Figure()
                        
                        # 添加收盘价
                        close_values = recent_data['close'].astype(float)
                        fig_price.add_trace(go.Scatter(
                            x=recent_data['trade_date_display'],
                            y=close_values,
                            mode='lines+markers',
                            name='收盘价',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4)
                        ))
                        
                        # 添加可用的MA线
                        ma_lines = {'MA5': 'orange', 'MA10': 'green', 'MA20': 'red'}
                        for ma_col, color in ma_lines.items():
                            if ma_col in recent_data.columns and recent_data[ma_col].notna().any():
                                fig_price.add_trace(go.Scatter(
                                    x=recent_data['trade_date_display'],
                                    y=recent_data[ma_col].astype(float),
                                    mode='lines',
                                    name=ma_col,
                                    line=dict(color=color, width=1, dash='dash')
                                ))
                        
                        fig_price.update_layout(
                            title=f'{stock_name} 价格走势与移动平均线',
                            xaxis_title='日期',
                            yaxis_title='价格',
                            height=300,
                            hovermode='x unified',
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01
                            )
                        )
                        st.plotly_chart(fig_price, use_container_width=True)
                        
                        # 如果计算了MACD，显示MACD图表
                        if all(col in recent_data.columns for col in ['DIF', 'DEA', 'MACD']):
                            fig_macd = go.Figure()
                            
                            fig_macd.add_trace(go.Scatter(
                                x=recent_data['trade_date_display'],
                                y=recent_data['DIF'].astype(float),
                                mode='lines',
                                name='DIF',
                                line=dict(color='blue', width=2)
                            ))
                            
                            fig_macd.add_trace(go.Scatter(
                                x=recent_data['trade_date_display'],
                                y=recent_data['DEA'].astype(float),
                                mode='lines',
                                name='DEA',
                                line=dict(color='red', width=2)
                            ))
                            
                            # MACD柱状图
                            macd_values = recent_data['MACD'].astype(float)
                            colors = ['green' if val >= 0 else 'red' for val in macd_values]
                            
                            fig_macd.add_trace(go.Bar(
                                x=recent_data['trade_date_display'],
                                y=macd_values,
                                name='MACD',
                                marker_color=colors,
                                opacity=0.6
                            ))
                            
                            fig_macd.update_layout(
                                title=f'{stock_name} MACD指标',
                                xaxis_title='日期',
                                yaxis_title='MACD值',
                                height=250,
                                hovermode='x unified',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            st.plotly_chart(fig_macd, use_container_width=True)
                else:
                    st.info("技术指标计算失败，无可用数据")
            
            # 选择股票进行计算
            if 'eval_result_tb2' in locals() and not eval_result_tb2.empty:
                # 获取可选的股票列表（限制在前20只，避免列表太长）
                available_stocks = eval_result_tb2['股票简称'].head(20).tolist()
                
                if available_stocks:
                    selected_stocks = st.multiselect(
                        "选择股票进行技术指标计算",
                        options=available_stocks,
                        default=available_stocks[:2] if len(available_stocks) >= 2 else available_stocks,
                        key=f'tech_stocks_{nm}'
                    )
                    
                    if selected_stocks:
                        # 获取选中的股票代码
                        selected_codes = eval_result_tb2[eval_result_tb2['股票简称'].isin(selected_stocks)]['股票代码'].tolist()
                        
                        # 4.2 指标计算结果展示
                        st.markdown("##### 📊 指标计算结果")
                        
                        # 创建标签页展示不同股票的技术指标
                        tech_tabs = st.tabs(selected_stocks)
                        
                        # 读取交易数据
                        try:
                            # 尝试读取多个年份的复权交易数据
                            trdata_files = ['复权交易数据2023.csv', '复权交易数据2024.csv', '复权交易数据2025.csv']
                            trdata_list = []
                            
                            for file in trdata_files:
                                try:
                                    data = pd.read_csv(file)
                                    if not data.empty:
                                        trdata_list.append(data)
                                except FileNotFoundError:
                                    st.warning(f"未找到文件: {file}")
                                except Exception as e:
                                    st.warning(f"读取 {file} 失败: {e}")
                            
                            if trdata_list:
                                trdata = pd.concat(trdata_list, ignore_index=True)
                                    
                            else:
                                st.error("未能读取任何交易数据文件")
                                trdata = pd.DataFrame()
                                
                        except Exception as e:
                            st.error(f"读取交易数据失败: {str(e)}")
                            trdata = pd.DataFrame()
                        
                        for idx, stock_name in enumerate(selected_stocks):
                            with tech_tabs[idx]:
                                try:
                                    # 获取股票代码
                                    stock_row = eval_result_tb2[eval_result_tb2['股票简称'] == stock_name]
                                    if stock_row.empty:
                                        st.info(f"未找到股票 '{stock_name}' 的信息")
                                        continue
                                    
                                    stock_code = stock_row['股票代码'].iloc[0]
                                    st.write(f"**股票代码**: {stock_code}")
                                    
                                    if trdata.empty:
                                        st.info("交易数据为空，无法计算技术指标")
                                        continue
                                    
                                    # 在交易数据中查找该股票
                                    stock_data = trdata[trdata['ts_code'] == stock_code].copy()
                                    
                                    if stock_data.empty:
                                        st.info(f"未找到股票 {stock_code} 在交易数据中")
                                        st.write("尝试搜索其他格式...")
                                        # 尝试去掉后缀搜索
                                        base_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
                                        matching_codes = [code for code in trdata['ts_code'].unique() 
                                                       if isinstance(code, str) and code.startswith(base_code)]
                                        
                                        if matching_codes:
                                            st.write(f"找到可能的匹配代码: {matching_codes[:3]}")
                                            stock_data = trdata[trdata['ts_code'].isin(matching_codes)].copy()
                                    
                                    if not stock_data.empty:                                        
                                        # 处理数据
                                        if 'trade_date' in stock_data.columns:
                                            # 检查日期格式
                                            first_date = str(stock_data['trade_date'].iloc[0])
                                            st.write(f"原始日期示例: {first_date}")
                                            
                                            # 尝试作为整数处理（YYYYMMDD格式）
                                            try:
                                                stock_data['trade_date'] = pd.to_datetime(
                                                    stock_data['trade_date'].astype(str), 
                                                    format='%Y%m%d',
                                                    errors='coerce'
                                                )
                                                
                                                # 检查是否有无效日期
                                                invalid_dates = stock_data['trade_date'].isna().sum()
                                                if invalid_dates > 0:
                                                    st.warning(f"有 {invalid_dates} 条记录的日期转换失败")
                                                
                                            except Exception as e:
                                                st.warning(f"日期转换失败: {e}")
                                                # 保持原样
                                                stock_data['trade_date'] = stock_data['trade_date'].astype(str)
                                        
                                        # 2. 确保数值列正确
                                        numeric_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
                                        for col in numeric_cols:
                                            if col in stock_data.columns:
                                                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                                        
                                        # 3. 排序
                                        stock_data = stock_data.sort_values('trade_date')
                                        
                                        # 4. 检查数据质量
                                        st.write(f"- 有效收盘价记录数: {stock_data['close'].notna().sum()}")
                                        
                                        missing_cols = [col for col in ['open', 'high', 'low'] if col not in stock_data.columns]
                                        if missing_cols:
                                            st.warning(f"缺少列: {missing_cols}")
                                            # 使用收盘价填充缺失的价格列
                                            if 'close' in stock_data.columns:
                                                for col in missing_cols:
                                                    if col in ['open', 'high', 'low']:
                                                        stock_data[col] = stock_data['close']
                                        
                                        # 5. 计算技术指标
                                        tech_data = calculate_technical_indicators_robust(stock_data)
                                        
                                        # 6. 显示技术指标
                                        display_technical_indicators(tech_data, stock_name)
                                        
                                    else:
                                        st.info(f"未找到股票 {stock_code} 的交易数据")
                                        st.write("交易数据中的股票代码示例:", trdata['ts_code'].unique()[:10])
                                        
                                except Exception as e:
                                    st.error(f"处理股票 {stock_name} 时出错: {str(e)}")
                                    import traceback
                                    st.write(traceback.format_exc())
                                    
                    else:
                        st.info("请选择股票进行技术指标计算")
                else:
                    st.info("暂无可用的股票列表")
            else:
                st.info("请先进行综合评价分析以获取股票排名")
                
            # 5. 预测模型构建 ==========
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### 🤖 预测模型构建")
            
            # 5.1 数据集划分说明
            st.markdown("""
            **数据集处理：**
            - **数据集划分**: 训练集（70%）、测试集（20%）、预测集（10%）
            - **数据预处理**: 缺失值处理、标准化、特征工程
            """)
            
            # 选择一只股票进行预测
            if 'selected_stocks' in locals() and selected_stocks:
                predict_stock = st.selectbox("选择股票进行价格趋势预测", selected_stocks, key=f'predict_stock_{nm}')
                
                if predict_stock:
                    try:
                        # 获取股票代码
                        stock_row = eval_result_tb2[eval_result_tb2['股票简称'] == predict_stock]
                        if stock_row.empty:
                            st.info(f"未找到股票 '{predict_stock}' 的信息")
                            st.stop()
                        
                        stock_code = stock_row['股票代码'].iloc[0]
                        st.write(f"**预测股票**: {predict_stock} ({stock_code})")
                        
                        # 读取交易数据
                        try:
                            # 尝试读取复权交易数据
                            trdata_files = ['复权交易数据2023.csv', '复权交易数据2024.csv', '复权交易数据2025.csv']
                            stock_data_all = []
                            
                            for file in trdata_files:
                                try:
                                    data = pd.read_csv(file)
                                    if not data.empty and 'ts_code' in data.columns:
                                        # 筛选该股票的数据
                                        stock_data = data[data['ts_code'] == stock_code].copy()
                                        if not stock_data.empty:
                                            stock_data_all.append(stock_data)
                                except:
                                    continue
                            
                            if not stock_data_all:
                                st.error("未找到该股票的交易数据")
                                st.stop()
                            
                            # 合并数据
                            stock_data = pd.concat(stock_data_all, ignore_index=True)
                            
                            if stock_data.empty:
                                st.error("合并后的数据为空")
                                st.stop()

                        except Exception as e:
                            st.error(f"读取交易数据失败: {str(e)}")
                            st.stop()
                        
                        # 数据预处理
                        st.markdown("##### 🔧 数据预处理")
                        
                        # 1. 处理日期
                        if 'trade_date' in stock_data.columns:
                            try:
                                # 转换为日期格式
                                stock_data['trade_date'] = pd.to_datetime(
                                    stock_data['trade_date'].astype(str), 
                                    format='%Y%m%d',
                                    errors='coerce'
                                )
                                stock_data = stock_data.sort_values('trade_date')
                            except:
                                stock_data = stock_data.sort_values('trade_date')
                        
                        # 2. 检查必要的列
                        required_cols = ['close']
                        optional_cols = ['open', 'high', 'low', 'vol']
                                                
                        # 3. 处理缺失值-对于缺失的价格列，使用收盘价填充
                        price_cols = ['open', 'high', 'low', 'close']
                        for col in price_cols:
                            if col in stock_data.columns:
                                # 填充缺失值
                                stock_data[col] = stock_data[col].fillna(method='ffill').fillna(method='bfill')
                        
                        # 对于成交量，用平均值填充
                        if 'vol' in stock_data.columns:
                            stock_data['vol'] = stock_data['vol'].fillna(stock_data['vol'].mean())
                        
                        # 4. 计算技术指标
                        st.markdown("##### 📊 技术指标计算")
                        
                        def calculate_tech_indicators_for_prediction(df):
                            """为预测准备的技术指标计算（处理缺失列）"""
                            df = df.sort_values('trade_date').copy()
                            
                            # 确保有收盘价
                            if 'close' not in df.columns:
                                return df
                            
                            # 1. MA移动平均线
                            df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
                            df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
                            df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
                            
                            # 2. 价格变化率
                            df['price_change'] = df['close'].pct_change()
                            
                            # 3. 成交量变化率
                            if 'vol' in df.columns:
                                df['vol_change'] = df['vol'].pct_change()
                            
                            # 4. 简单特征
                            df['price_ma_ratio'] = df['close'] / df['MA5']
                            
                            # 5. 如果存在高低价，计算价格区间
                            if all(col in df.columns for col in ['high', 'low']):
                                df['price_range'] = (df['high'] - df['low']) / df['close']
                            else:
                                # 使用价格波动作为替代
                                df['price_range'] = df['close'].rolling(window=5).std() / df['close']
                            
                            # 6. 涨跌趋势（目标变量）
                            df['trend'] = (df['close'].shift(-1) > df['close']).astype(int)  # 预测下一日涨跌
                            
                            # 7. 移除最后一行（因为trend是NaN）
                            df = df[:-1] if len(df) > 1 else df
                            
                            return df
                        
                        # 计算技术指标
                        tech_data = calculate_tech_indicators_for_prediction(stock_data)
                        
                        if tech_data.empty:
                            st.error("技术指标计算失败，数据为空")
                            st.stop()
                        
                        # 显示计算的技术指标
                        display_cols = ['trade_date', 'close', 'MA5', 'MA10', 'MA20', 
                                      'price_change', 'price_range', 'trend']
                        display_cols = [col for col in display_cols if col in tech_data.columns]
                        
                        st.write("技术指标数据（前10行）：")
                        st.dataframe(tech_data[display_cols].head(10))
                        st.write(f"总数据量: {len(tech_data)} 条记录")
                        
                        # 5. 准备训练数据
                        st.markdown("##### 🎯 准备训练数据")
                        
                        # 选择特征列
                        feature_cols = ['MA5', 'MA10', 'MA20', 'price_change', 'price_range']
                        if 'vol_change' in tech_data.columns:
                            feature_cols.append('vol_change')
                        
                        # 确保特征列都存在
                        feature_cols = [col for col in feature_cols if col in tech_data.columns]
                        
                        # 提取特征和目标
                        X = tech_data[feature_cols].copy()
                        y = tech_data['trend'].copy()
                        
                        # 处理缺失值
                        X = X.fillna(0)
                        
                        if X.empty or y.empty:
                            st.error("特征或目标数据为空")
                            st.stop()
                        
                        # 数据集划分
                        st.markdown("##### 📊 数据集划分")
                        
                        total_samples = len(X)
                        train_size = int(total_samples * 0.7)
                        test_size = int(total_samples * 0.2)
                        pred_size = total_samples - train_size - test_size
                        
                        # 确保有足够的数据
                        if train_size < 10:
                            st.error(f"数据量太少 ({total_samples})，无法进行有效的模型训练")
                            st.stop()
                        
                        if pred_size < 1:
                            pred_size = 1
                            test_size = total_samples - train_size - pred_size
                        
                        st.write(f"- 训练集: {train_size} 个样本 ({train_size/total_samples:.1%})")
                        st.write(f"- 测试集: {test_size} 个样本 ({test_size/total_samples:.1%})")
                        st.write(f"- 预测集: {pred_size} 个样本 ({pred_size/total_samples:.1%})")
                        
                        # 划分数据集
                        X_train = X.iloc[:train_size]
                        y_train = y.iloc[:train_size]
                        
                        X_test = X.iloc[train_size:train_size+test_size]
                        y_test = y.iloc[train_size:train_size+test_size]
                        
                        X_pred = X.iloc[train_size+test_size:]
                        y_pred_actual = y.iloc[train_size+test_size:]
                        
                        # 数据集标签页
                        tb_1, tb_2, tb_3 = st.tabs(["训练集", "测试集", "预测数据集"])
                        
                        with tb_1:
                            st.markdown("##### 训练集数据")
                            train_df = X_train.copy()
                            train_df['标签'] = y_train.values
                            st.dataframe(train_df.head(20), use_container_width=True)
                            st.write(f"训练集样本数: {len(X_train)}")
                            
                            # 类别分布
                            train_class_dist = y_train.value_counts()
                            st.write("训练集类别分布:")
                            st.write(f"- 上涨 (1): {train_class_dist.get(1, 0)} 个样本")
                            st.write(f"- 下跌 (0): {train_class_dist.get(0, 0)} 个样本")
                        
                        with tb_2:
                            st.markdown("##### 测试集数据")
                            test_df = X_test.copy()
                            test_df['标签'] = y_test.values
                            st.dataframe(test_df.head(20), use_container_width=True)
                            st.write(f"测试集样本数: {len(X_test)}")
                        
                        with tb_3:
                            st.markdown("##### 预测数据集")
                            pred_df = X_pred.copy()
                            pred_df['实际标签'] = y_pred_actual.values
                            st.dataframe(pred_df, use_container_width=True)
                            st.write(f"预测集样本数: {len(X_pred)}")
                        
                        # 模型选择
                        st.markdown("##### 🔧 模型选择")
                        
                        model_type = st.selectbox(
                            "选择预测模型", 
                            ['逻辑回归', '随机森林', '支持向量机', '神经网络', '梯度提升树'],
                            help="选择用于价格趋势预测的模型",
                            key=f'model_select_{nm}'
                        )
                        
                        # 标准化
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        X_pred_scaled = scaler.transform(X_pred) if len(X_pred) > 0 else None
                        
                        # 训练模型
                        if st.button("开始训练模型", key=f'train_btn_{nm}', type="primary"):
                            with st.spinner("正在训练模型..."):
                                try:
                                    # 选择模型
                                    if model_type == '逻辑回归':
                                        from sklearn.linear_model import LogisticRegression
                                        model = LogisticRegression(random_state=42, max_iter=1000)
                                    elif model_type == '随机森林':
                                        from sklearn.ensemble import RandomForestClassifier
                                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                                    elif model_type == '支持向量机':
                                        from sklearn.svm import SVC
                                        model = SVC(random_state=42, probability=True)
                                    elif model_type == '神经网络':
                                        from sklearn.neural_network import MLPClassifier
                                        model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
                                    else:  # 梯度提升树
                                        from sklearn.ensemble import GradientBoostingClassifier
                                        model = GradientBoostingClassifier(random_state=42)
                                    
                                    # 训练模型
                                    model.fit(X_train_scaled, y_train)
                                    
                                    # 预测
                                    y_train_pred = model.predict(X_train_scaled)
                                    y_test_pred = model.predict(X_test_scaled)
                                    
                                    if X_pred_scaled is not None and len(X_pred_scaled) > 0:
                                        y_pred_pred = model.predict(X_pred_scaled)
                                    else:
                                        y_pred_pred = []
                                    
                                    # 计算准确率
                                    from sklearn.metrics import accuracy_score
                                    train_acc = accuracy_score(y_train, y_train_pred)
                                    test_acc = accuracy_score(y_test, y_test_pred)
                                    
                                    st.success("✅ 模型训练完成！")
                                    
                                    # 6. 展示预测结果 ==========
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    st.markdown("#### 📊 预测结果分析")
                                    
                                    # 显示准确率
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("训练集准确率", f"{train_acc:.2%}")
                                    with col2:
                                        st.metric("测试集准确率", f"{test_acc:.2%}")
                                    with col3:
                                        if len(y_pred_pred) > 0 and len(y_pred_actual) > 0:
                                            pred_acc = accuracy_score(y_pred_actual, y_pred_pred)
                                            st.metric("预测集准确率", f"{pred_acc:.2%}")
                                        else:
                                            st.metric("预测集样本数", len(y_pred_actual))
                                    
                                    # 预测结果对比
                                    if len(y_pred_pred) > 0:
                                        st.markdown("##### 预测数据集的实际值与预测值对比")
                                        
                                        pred_results = pd.DataFrame({
                                            '实际趋势': y_pred_actual.values,
                                            '预测趋势': y_pred_pred,
                                            '预测结果': ['✅ 正确' if a == p else '❌ 错误' for a, p in zip(y_pred_actual.values, y_pred_pred)]
                                        })
                                        
                                        st.dataframe(pred_results, use_container_width=True)
                                        
                                        # 计算预测准确率
                                        correct_predictions = (y_pred_actual.values == y_pred_pred).sum()
                                        total_predictions = len(y_pred_pred)
                                        
                                        st.write(f"预测准确率: {correct_predictions}/{total_predictions} = {correct_predictions/total_predictions:.2%}")
                                    
                                    # 7. 量化投资策略设计 ==========
                                    st.markdown("<hr>", unsafe_allow_html=True)
                                    st.markdown("#### 💹 量化投资策略设计")
                                    
                                    st.markdown("""
                                    **根据预测结果设计量化投资策略：**
                                    1. **买入信号**: 预测下一交易日上涨 (预测趋势=1)
                                    2. **卖出信号**: 预测下一交易日下跌 (预测趋势=0)  
                                    3. **持仓策略**: 每次全仓买入/卖出
                                    4. **交易成本**: 暂不考虑
                                    """)
                                    
                                    # 简单的策略回测
                                    if len(y_pred_pred) > 0:
                                        # 模拟策略收益
                                        initial_capital = 100000  # 初始资金10万
                                        capital = initial_capital
                                        positions = 0  # 持仓数量
                                        trade_history = []
                                        
                                        # 使用预测集的价格数据
                                        pred_prices = tech_data['close'].iloc[train_size+test_size:].values
                                        pred_dates = tech_data['trade_date'].iloc[train_size+test_size:].values
                                        
                                        for i in range(len(y_pred_pred)):
                                            if i >= len(pred_prices) - 1:
                                                break
                                            
                                            current_price = pred_prices[i]
                                            next_price = pred_prices[i+1]
                                            
                                            # 根据预测信号交易
                                            if y_pred_pred[i] == 1:  # 预测上涨，买入
                                                if positions == 0:  # 空仓则买入
                                                    positions = capital / current_price
                                                    capital = 0
                                                    trade_history.append({
                                                        '日期': pred_dates[i],
                                                        '操作': '买入',
                                                        '价格': current_price,
                                                        '持仓': positions,
                                                        '资金': capital
                                                    })
                                            else:  # 预测下跌，卖出
                                                if positions > 0:  # 有持仓则卖出
                                                    capital = positions * current_price
                                                    positions = 0
                                                    trade_history.append({
                                                        '日期': pred_dates[i],
                                                        '操作': '卖出',
                                                        '价格': current_price,
                                                        '持仓': positions,
                                                        '资金': capital
                                                    })
                                        
                                        # 计算最终收益
                                        if positions > 0 and len(pred_prices) > 0:
                                            final_price = pred_prices[-1]
                                            final_capital = positions * final_price
                                        else:
                                            final_capital = capital
                                        
                                        total_return = (final_capital - initial_capital) / initial_capital
                                        
                                        # 基准收益（买入持有策略）
                                        if len(pred_prices) >= 2:
                                            buy_hold_return = (pred_prices[-1] - pred_prices[0]) / pred_prices[0]
                                        else:
                                            buy_hold_return = 0
                                        
                                        # 显示策略结果
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("💰 策略总收益率", f"{total_return:.2%}")
                                        with col2:
                                            st.metric("📈 买入持有收益率", f"{buy_hold_return:.2%}")
                                        with col3:
                                            alpha = total_return - buy_hold_return
                                            st.metric("⚡ 超额收益", f"{alpha:+.2%}")
                                        
                                        # 显示交易历史
                                        if trade_history:
                                            st.markdown("##### 📋 交易历史")
                                            trade_df = pd.DataFrame(trade_history)
                                            st.dataframe(trade_df, use_container_width=True)
                                    else:
                                        st.info("预测集数据不足，无法进行策略回测")
                                    
                                except Exception as e:
                                    st.error(f"模型训练失败: {str(e)}")
                                    import traceback
                                    st.write(traceback.format_exc())
                    except Exception as e:
                        st.error(f"预测模型构建出错: {str(e)}")
                        import traceback
                        st.write(traceback.format_exc())
                else:
                    st.info("请选择股票进行预测")
            else:
                st.info("请先进行技术指标计算")

            # 预测结果
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # 创建三个标签页来组织内容
            result_tab1, result_tab2, result_tab3 = st.tabs([
                "📊 预测结果分析", 
                "💹 量化投资策略设计", 
                "🤖 AI大模型解读与分析"
            ])
            
            with result_tab1:
                st.subheader('📊 预测结果分析')
                
                # 检查是否有训练好的模型
                if 'model' in locals() and model is not None:
                    # 检查是否有预测数据
                    if 'X_pred' in locals() and X_pred is not None and len(X_pred) > 0:
                        # 获取预测结果
                        if 'scaler' in locals():
                            X_pred_scaled = scaler.transform(X_pred)
                            y_pred = model.predict(X_pred_scaled)
                        else:
                            y_pred = model.predict(X_pred)
                        
                        # 获取实际值（如果有的话）
                        y_actual = None
                        if 'y_pred_actual' in locals() and y_pred_actual is not None:
                            y_actual = y_pred_actual
                        elif 'y_test' in locals() and y_test is not None:
                            # 如果没有单独的预测集，使用测试集
                            y_actual = y_test
                            if 'X_test' in locals() and X_test is not None:
                                if 'scaler' in locals():
                                    X_test_scaled = scaler.transform(X_test)
                                    y_pred = model.predict(X_test_scaled)
                                else:
                                    y_pred = model.predict(X_test)
                        
                        if y_actual is not None and y_pred is not None:
                            # 确保是 numpy 数组而不是 pandas Series
                            if hasattr(y_actual, 'values'):
                                y_actual_array = y_actual.values
                            else:
                                y_actual_array = np.array(y_actual)
                            
                            if hasattr(y_pred, 'flatten'):
                                y_pred_array = y_pred.flatten()
                            else:
                                y_pred_array = np.array(y_pred)
                            
                            min_len = min(len(y_actual_array), len(y_pred_array))
                            
                            # 创建对比DataFrame - 修复索引问题
                            comparison_data = []
                            for i in range(min_len):
                                # 确保能安全获取值
                                if i < len(y_actual_array) and i < len(y_pred_array):
                                    actual_val = int(y_actual_array[i])
                                    pred_val = int(y_pred_array[i])
                                    comparison_data.append({
                                        '序号': i + 1,
                                        '实际趋势': '上涨' if actual_val == 1 else '下跌',
                                        '预测趋势': '上涨' if pred_val == 1 else '下跌',
                                        '预测结果': '✅ 正确' if actual_val == pred_val else '❌ 错误'
                                    })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # 显示对比表格
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # 计算预测准确率
                            correct_predictions = 0
                            for i in range(min_len):
                                if y_actual_array[i] == y_pred_array[i]:
                                    correct_predictions += 1
                            
                            total_predictions = min_len
                            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                            
                            # 显示统计信息
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("总预测样本数", total_predictions)
                            with col2:
                                st.metric("正确预测数", correct_predictions)
                            with col3:
                                st.metric("预测准确率", f"{accuracy:.2%}")
                            
                            # 混淆矩阵
                            st.markdown("##### 预测结果混淆矩阵")
                            
                            try:
                                from sklearn.metrics import confusion_matrix
                                import plotly.figure_factory as ff
                                
                                cm = confusion_matrix(y_actual_array[:min_len], y_pred_array[:min_len])
                                
                                fig = ff.create_annotated_heatmap(
                                    z=cm,
                                    x=['预测下跌', '预测上涨'],
                                    y=['实际下跌', '实际上涨'],
                                    colorscale='Blues',
                                    showscale=True
                                )
                                
                                fig.update_layout(
                                    title='预测结果混淆矩阵',
                                    xaxis_title='预测值',
                                    yaxis_title='实际值',
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.info(f"混淆矩阵生成失败: {e}")
                            
                            # 预测结果可视化
                            st.markdown("##### 预测结果趋势图")
                            
                            # 创建时间序列对比图
                            try:
                                fig = go.Figure()
                                
                                # 添加实际趋势线
                                fig.add_trace(go.Scatter(
                                    x=list(range(min_len)),
                                    y=y_actual_array[:min_len],
                                    mode='lines+markers',
                                    name='实际趋势',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=6)
                                ))
                                
                                # 添加预测趋势线
                                fig.add_trace(go.Scatter(
                                    x=list(range(min_len)),
                                    y=y_pred_array[:min_len],
                                    mode='lines+markers',
                                    name='预测趋势',
                                    line=dict(color='red', width=2, dash='dash'),
                                    marker=dict(size=6)
                                ))
                                
                                fig.update_layout(
                                    title='实际趋势 vs 预测趋势对比',
                                    xaxis_title='样本序号',
                                    yaxis_title='趋势 (1=上涨, 0=下跌)',
                                    yaxis=dict(tickvals=[0, 1], ticktext=['下跌', '上涨']),
                                    height=400,
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.info(f"趋势图生成失败: {e}")
                                
                            # 添加分类报告
                            try:
                                from sklearn.metrics import classification_report
                                
                                st.markdown("##### 详细分类报告")
                                
                                # 生成分类报告
                                report = classification_report(
                                    y_actual_array[:min_len], 
                                    y_pred_array[:min_len],
                                    target_names=['下跌', '上涨'],
                                    output_dict=True
                                )
                                
                                # 转换为DataFrame
                                report_df = pd.DataFrame(report).transpose()
                                
                                # 格式化显示
                                def format_percent(val):
                                    if isinstance(val, (int, float)):
                                        return f"{val:.2%}"
                                    return val
                                
                                st.dataframe(report_df.style.format({
                                    'precision': '{:.2%}',
                                    'recall': '{:.2%}',
                                    'f1-score': '{:.2%}'
                                }), use_container_width=True)
                                
                            except Exception as e:
                                st.info(f"分类报告生成失败: {e}")
                                
                        else:
                            st.info("暂无预测数据")
                    else:
                        st.info("请先完成模型训练以查看预测结果")
                else:
                    st.info("请先训练模型以获取预测结果")
            
            with result_tab2:
                st.subheader('💹 量化投资策略设计')
                
                st.markdown("""
                **策略逻辑：**
                1. **买入信号**: 模型预测下一交易日上涨
                2. **卖出信号**: 模型预测下一交易日下跌
                3. **持仓策略**: 全仓操作
                4. **初始资金**: 100,000元
                """)
                
                # 获取股票数据
                if 'selected_stocks' in locals() and selected_stocks:
                    # 选择要回测的股票
                    backtest_stock = st.selectbox(
                        "选择股票进行策略回测",
                        selected_stocks,
                        key='backtest_stock'
                    )
                    
                    if backtest_stock and 'eval_result_tb2' in locals():
                        try:
                            # 获取股票代码
                            stock_row = eval_result_tb2[eval_result_tb2['股票简称'] == backtest_stock]
                            if not stock_row.empty:
                                stock_code = stock_row['股票代码'].iloc[0]
                                
                                # 读取该股票的交易数据
                                trdata_files = ['复权交易数据2023.csv', '复权交易数据2024.csv', '复权交易数据2025.csv']
                                all_stock_data = []
                                
                                for file in trdata_files:
                                    try:
                                        data = pd.read_csv(file)
                                        # 支持多种代码格式匹配
                                        if 'ts_code' in data.columns:
                                            # 尝试精确匹配
                                            exact_match = data[data['ts_code'] == stock_code].copy()
                                            if not exact_match.empty:
                                                all_stock_data.append(exact_match)
                                            else:
                                                # 尝试部分匹配（去掉后缀）
                                                if isinstance(stock_code, str):
                                                    base_code = stock_code.split('.')[0] if '.' in stock_code else stock_code
                                                    partial_match = data[data['ts_code'].str.contains(str(base_code), na=False)].copy()
                                                    if not partial_match.empty:
                                                        all_stock_data.append(partial_match)
                                    except Exception as e:
                                        st.warning(f"读取 {file} 失败: {e}")
                                        continue
                                
                                if all_stock_data:
                                    stock_data = pd.concat(all_stock_data, ignore_index=True)
                                    
                                    # 显示数据预览
                                    st.info(f"找到 {len(stock_data)} 条交易数据")
                                    
                                    # 策略参数
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        initial_capital = st.number_input(
                                            "初始资金 (元)", 
                                            value=100000,
                                            min_value=10000,
                                            max_value=1000000,
                                            step=10000,
                                            key='capital_input'
                                        )
                                    
                                    with col2:
                                        # 获取预测结果（如果模型已训练）
                                        use_model_signal = st.checkbox("使用模型预测信号", value=True, key='model_signal_check')
                                    
                                    if st.button("执行策略回测", type="primary", key='backtest_btn'):
                                        with st.spinner("正在执行策略回测..."):
                                            # 准备数据
                                            stock_data = stock_data.sort_values('trade_date')
                                            stock_data = stock_data.reset_index(drop=True)
                                            
                                            # 显示数据信息
                                            st.write(f"数据时间范围: {stock_data['trade_date'].iloc[0]} 至 {stock_data['trade_date'].iloc[-1]}")
                                            st.write(f"数据条数: {len(stock_data)}")
                                            
                                            # 确保有必要的列
                                            required_cols = ['close', 'open', 'high', 'low']
                                            missing_cols = [col for col in required_cols if col not in stock_data.columns]
                                            
                                            if missing_cols:
                                                st.error(f"交易数据缺少必要列: {missing_cols}")
                                                # 尝试重命名列
                                                rename_map = {}
                                                if '收盘价' in stock_data.columns and 'close' not in stock_data.columns:
                                                    rename_map['收盘价'] = 'close'
                                                if '开盘价' in stock_data.columns and 'open' not in stock_data.columns:
                                                    rename_map['开盘价'] = 'open'
                                                if '最高价' in stock_data.columns and 'high' not in stock_data.columns:
                                                    rename_map['最高价'] = 'high'
                                                if '最低价' in stock_data.columns and 'low' not in stock_data.columns:
                                                    rename_map['最低价'] = 'low'
                                                
                                                if rename_map:
                                                    stock_data = stock_data.rename(columns=rename_map)
                                                    st.success(f"已重命名列: {rename_map}")
                                            
                                            if 'close' not in stock_data.columns:
                                                st.error("交易数据中没有收盘价信息")
                                                st.stop()
                                            
                                            # 获取价格数据
                                            prices = stock_data['close'].values
                                            st.write(f"价格数据长度: {len(prices)}")
                                            
                                            # 检查价格数据有效性
                                            if len(prices) < 10:
                                                st.error(f"价格数据不足 ({len(prices)} 条)，至少需要10条数据进行回测")
                                                st.stop()
                                            
                                            # 检查价格数据是否有NaN
                                            nan_count = np.isnan(prices).sum()
                                            if nan_count > 0:
                                                st.warning(f"价格数据中有 {nan_count} 个NaN值，将进行填充")
                                                prices = pd.Series(prices).fillna(method='ffill').fillna(method='bfill').values
                                            
                                            # 使用简单策略或模型策略
                                            signals = None
                                            
                                            if use_model_signal and 'model' in locals() and model is not None:
                                                try:
                                                    # 计算技术指标
                                                    tech_data = calculate_technical_indicators_robust(stock_data)
                                                    tech_data = tech_data.dropna()
                                                    
                                                    if not tech_data.empty and len(tech_data) >= len(prices):
                                                        # 准备特征
                                                        feature_cols = ['MA5', 'MA10', 'MA20', 'price_change', 'price_range']
                                                        available_cols = [col for col in feature_cols if col in tech_data.columns]
                                                        
                                                        if available_cols:
                                                            X = tech_data[available_cols].fillna(0)
                                                            
                                                            # 确保X的长度与价格匹配
                                                            if len(X) != len(prices):
                                                                st.warning(f"特征数据长度 ({len(X)}) 与价格长度 ({len(prices)}) 不匹配，使用简单策略")
                                                                use_model_signal = False
                                                            else:
                                                                # 预测信号
                                                                try:
                                                                    if 'scaler' in locals():
                                                                        X_scaled = scaler.transform(X)
                                                                        signals = model.predict(X_scaled)
                                                                    else:
                                                                        signals = model.predict(X)

                                                                    st.write(f"信号分布: 买入信号 {np.sum(signals==1)}, 卖出信号 {np.sum(signals==0)}")
                                                                except Exception as e:
                                                                    st.warning(f"模型预测失败: {e}，使用简单策略")
                                                                    use_model_signal = False
                                                        else:
                                                            st.warning("技术指标计算不完整，使用简单策略")
                                                            use_model_signal = False
                                                    else:
                                                        st.warning("技术指标计算失败，使用简单策略")
                                                        use_model_signal = False
                                                        
                                                except Exception as e:
                                                    st.warning(f"模型信号生成失败: {e}，使用简单策略")
                                                    use_model_signal = False
                                            
                                            # 如果不使用模型或模型失败，使用简单策略
                                            if not use_model_signal or signals is None:
                                                # 简单策略：当价格上涨时买入，下跌时卖出
                                                if len(prices) > 1:
                                                    price_changes = np.diff(prices)
                                                    signals = np.zeros(len(prices))
                                                    # 今天的信号基于明天的价格变化
                                                    signals[:-1] = (price_changes > 0).astype(int)
                                                    st.write(f"信号分布: 买入信号 {np.sum(signals==1)}, 卖出信号 {np.sum(signals==0)}")
                                                else:
                                                    st.error("价格数据不足，无法生成简单策略信号")
                                                    st.stop()
                                            
                                            # 确保信号长度与价格匹配
                                            if len(signals) != len(prices):
                                                st.warning(f"信号长度 ({len(signals)}) 与价格长度 ({len(prices)}) 不匹配，进行调整")
                                                min_len = min(len(signals), len(prices))
                                                signals = signals[:min_len]
                                                prices = prices[:min_len]
                                            
                                            # 执行回测
                                            capital = initial_capital
                                            positions = 0
                                            trade_history = []
                                            portfolio_values = []
                                            
                                            for i in range(len(prices)):
                                                current_price = prices[i]
                                                
                                                # 记录资产价值
                                                current_value = capital + (positions * current_price)
                                                portfolio_values.append(current_value)
                                                
                                                # 交易信号（从第二天开始交易）
                                                if i > 0:
                                                    signal = signals[i]
                                                    
                                                    # 执行交易
                                                    if signal == 1 and positions == 0 and capital > 0:
                                                        # 买入
                                                        positions = capital / current_price
                                                        capital = 0
                                                        
                                                        trade_date = stock_data.iloc[i]['trade_date'] if 'trade_date' in stock_data.columns else f"Day {i+1}"
                                                        trade_history.append({
                                                            '日期': trade_date,
                                                            '操作': '买入',
                                                            '价格': round(current_price, 2),
                                                            '仓位': round(positions, 2),
                                                            '总资产': round(current_value, 2)
                                                        })
                                                    
                                                    elif signal == 0 and positions > 0:
                                                        # 卖出
                                                        capital = positions * current_price
                                                        positions = 0
                                                        
                                                        trade_date = stock_data.iloc[i]['trade_date'] if 'trade_date' in stock_data.columns else f"Day {i+1}"
                                                        trade_history.append({
                                                            '日期': trade_date,
                                                            '操作': '卖出',
                                                            '价格': round(current_price, 2),
                                                            '仓位': 0,
                                                            '总资产': round(current_value, 2)
                                                        })
                                            
                                            # 计算最终收益
                                            final_price = prices[-1]
                                            if positions > 0:
                                                final_value = positions * final_price
                                            else:
                                                final_value = capital
                                            
                                            total_return = (final_value - initial_capital) / initial_capital
                                            
                                            # 计算买入持有收益
                                            if len(prices) > 1:
                                                buy_hold_return = (prices[-1] - prices[0]) / prices[0]
                                            else:
                                                buy_hold_return = 0
                                            
                                            # 绘制资产曲线
                                            fig = go.Figure()
                                            
                                            # 策略资产曲线
                                            fig.add_trace(go.Scatter(
                                                x=list(range(len(portfolio_values))),
                                                y=portfolio_values,
                                                mode='lines',
                                                name='策略资产',
                                                line=dict(color='green', width=2)
                                            ))
                                            
                                            # 买入持有资产曲线
                                            if len(prices) > 1:
                                                bh_values = [initial_capital * (1 + (prices[min(i, len(prices)-1)] - prices[0]) / prices[0]) 
                                                            for i in range(len(portfolio_values))]
                                                fig.add_trace(go.Scatter(
                                                    x=list(range(len(portfolio_values))),
                                                    y=bh_values,
                                                    mode='lines',
                                                    name='买入持有',
                                                    line=dict(color='blue', width=2, dash='dash')
                                                ))
                                            
                                            fig.update_layout(
                                                title=f'{backtest_stock} 策略表现对比',
                                                xaxis_title='交易日',
                                                yaxis_title='资产价值 (元)',
                                                height=400,
                                                showlegend=True
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                            
                                            # 显示交易历史
                                            if trade_history:
                                                st.markdown("##### 📋 交易历史")
                                                trade_df = pd.DataFrame(trade_history)
                                                st.dataframe(trade_df, use_container_width=True)
                                                
                                                # 统计
                                                buy_count = len([t for t in trade_history if t['操作'] == '买入'])
                                                sell_count = len([t for t in trade_history if t['操作'] == '卖出'])
                                                
                                                col1, col2, col3 = st.columns(3)
                                                with col1:
                                                    st.metric("买入次数", buy_count)
                                                with col2:
                                                    st.metric("卖出次数", sell_count)
                                                
                                                # 计算胜率
                                                if sell_count > 0:
                                                    profitable_trades = 0
                                                    total_return_rate = 0
                                                    for i in range(1, len(trade_history)):
                                                        if trade_history[i]['操作'] == '卖出' and i > 0 and trade_history[i-1]['操作'] == '买入':
                                                            buy_price = trade_history[i-1]['价格']
                                                            sell_price = trade_history[i]['价格']
                                                            trade_return = (sell_price - buy_price) / buy_price
                                                            total_return_rate += trade_return
                                                            if sell_price > buy_price:
                                                                profitable_trades += 1
                                                    win_rate = profitable_trades / sell_count if sell_count > 0 else 0
                                                    avg_return = total_return_rate / sell_count if sell_count > 0 else 0
                                                    
                                                    with col3:
                                                        st.metric("交易胜率", f"{win_rate:.2%}")
                                                    
                                                    st.write(f"平均每笔交易收益率: {avg_return:.2%}")
                                            
                                            else:
                                                st.info("本次回测期间没有发生交易")
                                        
                                else:
                                    st.warning(f"未找到股票 {backtest_stock} ({stock_code}) 的交易数据")
                            else:
                                st.warning("未找到该股票信息")
                                
                        except Exception as e:
                            st.error(f"策略回测失败: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.info("请先选择股票进行分析")
                                
            with result_tab3:
                st.subheader('🤖 AI大模型解读与分析')
                # 检查是否有选中的股票
                if 'selected_stocks' in locals() and selected_stocks:
                    # 股票选择
                    ai_stock = st.selectbox(
                        "选择要分析的股票",
                        selected_stocks,
                        key='ai_stock_select'
                    )
                    
                    if ai_stock and 'eval_result_tb2' in locals():
                        try:
                            # 获取股票信息
                            stock_info = eval_result_tb2[eval_result_tb2['股票简称'] == ai_stock]
                            if not stock_info.empty:
                                stock_code = stock_info['股票代码'].iloc[0]
                                
                                # 可以添加更多股票信息传递给AI
                                stock_price = stock_info['股价'].iloc[0] if '股价' in stock_info.columns else "未知"
                                pe_ratio = stock_info['市盈率'].iloc[0] if '市盈率' in stock_info.columns else "未知"
                                
                                # AI分析按钮
                                if st.button("开始AI分析", type="primary", key='ai_analyze_btn'):
                                    with st.spinner("DeepSeek AI正在分析中..."):
                                        import requests
                                        
                                        # 硅基流动平台的API配置
                                        api_key = "填写您的API密钥"  # API密钥
                                        
                                        # 硅基流动的DeepSeek API端点
                                        api_url = "https://api.siliconflow.cn/v1/chat/completions"
                                        
                                        headers = {
                                            "Authorization": f"Bearer {api_key}",
                                            "Content-Type": "application/json"
                                        }
                                        
                                        # 构建更详细的分析请求
                                        prompt = f"""作为专业的股票分析师，请分析以下股票：
                                                股票名称：{ai_stock}
                                                股票代码：{stock_code}
                                                当前股价：{stock_price}
                                                市盈率：{pe_ratio}
                                                请从以下几个方面进行分析：
                                                1. 基本面分析（财务状况、盈利能力、成长性）
                                                2. 技术面分析（趋势、支撑阻力位）
                                                3. 行业地位和竞争优势
                                                4. 风险提示
                                                5. 投资建议（短期、中期、长期）
                                                请提供详细、专业的分析，用中文回答。"""
                                        
                                        data = {
                                            "model": "deepseek-ai/DeepSeek-OCR",  # 硅基流动上的DeepSeek模型
                                            "messages": [
                                                {
                                                    "role": "system", 
                                                    "content": "你是一位资深的证券投资顾问，具有多年A股市场分析经验。请提供专业、客观的股票分析。"
                                                },
                                                {
                                                    "role": "user", 
                                                    "content": prompt
                                                }
                                            ],
                                            "temperature": 0.7,
                                            "max_tokens": 1500,
                                            "stream": False
                                        }
                                        
                                        try:
                                            response = requests.post(
                                                api_url,
                                                headers=headers,
                                                json=data,
                                                timeout=60  # 增加超时时间
                                            )
                                            
                                            if response.status_code == 200:
                                                result = response.json()
                                                ai_analysis = result['choices'][0]['message']['content']
                                                
                                                # 美化显示
                                                st.success("✅ AI分析完成！")
                                                st.markdown("### 📊 AI分析报告")
                                                st.markdown("---")
                                                st.markdown(ai_analysis)
                                                
                                            else:
                                                error_msg = f"API调用失败: {response.status_code}"
                                                try:
                                                    error_detail = response.json()
                                                    error_msg += f"\n错误详情: {error_detail}"
                                                except:
                                                    error_msg += f"\n响应内容: {response.text}"
                                                st.error(error_msg)
                                                
                                        except requests.exceptions.Timeout:
                                            st.error("请求超时，请稍后重试")
                                        except Exception as e:
                                            st.error(f"API调用异常: {str(e)}")
                            else:
                                st.warning("未找到该股票信息")
                        except Exception as e:
                            st.error(f"AI分析准备失败: {str(e)}")
                else:
                    st.info("请先选择股票进行AI分析")
# 执行函数
if __name__ == "__main__":
    st_fig()
