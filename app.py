import streamlit as st
import pandas as pd
import fcsparser
import tempfile
import os
import numpy as np
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.transform import transform
from scipy import stats

st.title("FACS Data Analysis")
st.write("FlowCytometry Standard（.fcs）ファイルからイベントデータを抽出し、CSV形式でダウンロードできるアプリケーションです。")

# ファイルアップロード
uploaded_file = st.file_uploader("FCS ファイルをアップロードしてください", type=['fcs'])

if uploaded_file is not None:
    try:
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # FCSファイルを読み込み
        with st.spinner('FCSファイルを処理中...'):
            meta, data = fcsparser.parse(tmp_file_path, reformat_meta=True)
        
        # データフレームに変換
        df = pd.DataFrame(data)
        
        # 一時ファイルを削除
        os.unlink(tmp_file_path)
        
        st.success(f"FCSファイルが正常に読み込まれました。{len(df)} イベントが検出されました。")
        
        # データのプレビュー表示
        st.subheader("データプレビュー（上位10行）")
        st.dataframe(df.head(10))
        
        # 散布図作成セクション
        st.subheader("散布図作成")
        
        # 数値列のみを取得
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X軸を選択してください", numeric_columns, key="x_axis")
            
            with col2:
                y_axis = st.selectbox("Y軸を選択してください", 
                                    [col for col in numeric_columns if col != x_axis], 
                                    key="y_axis")
            
            # プロットオプション
            st.write("**プロットオプション**")
            plot_col1, plot_col2, plot_col3 = st.columns(3)
            
            with plot_col1:
                sample_size = st.slider("表示するデータポイント数", 
                                      min_value=1000, 
                                      max_value=min(len(df), 50000), 
                                      value=min(10000, len(df)),
                                      step=1000)
            
            with plot_col2:
                color_by = st.selectbox("色分けする軸（オプション）", 
                                      ["なし"] + numeric_columns, 
                                      key="color_axis")
            
            with plot_col3:
                plot_width = st.slider("プロット幅", 400, 800, 600)
                plot_height = st.slider("プロット高さ", 300, 600, 400)
            
            # プロット作成ボタン
            if st.button("散布図を作成", type="primary"):
                with st.spinner('散布図を作成中...'):
                    # データをサンプリング
                    if len(df) > sample_size:
                        df_sample = df.sample(n=sample_size, random_state=42)
                    else:
                        df_sample = df.copy()
                    
                    # Bokeh散布図の作成
                    p = figure(width=plot_width, height=plot_height,
                             title=f"{y_axis} vs {x_axis}",
                             x_axis_label=x_axis,
                             y_axis_label=y_axis,
                             tools="pan,wheel_zoom,box_zoom,reset,save")
                    
                    # ホバーツールの設定
                    hover = HoverTool(tooltips=[
                        ("Index", "$index"),
                        (x_axis, f"@{x_axis}{{0.00}}"),
                        (y_axis, f"@{y_axis}{{0.00}}")
                    ])
                    
                    if color_by != "なし":
                        # カラーマッピングの設定
                        color_mapper = LinearColorMapper(palette=Viridis256, 
                                                       low=df_sample[color_by].min(), 
                                                       high=df_sample[color_by].max())
                        
                        # カラーバー付きの散布図
                        scatter = p.circle(x=x_axis, y=y_axis, 
                                         size=6, alpha=0.6,
                                         color=transform(color_by, color_mapper),
                                         source=df_sample)
                        
                        # カラーバーを追加
                        color_bar = ColorBar(color_mapper=color_mapper, 
                                           label_standoff=12,
                                           location=(0,0),
                                           title=color_by)
                        p.add_layout(color_bar, 'right')
                        
                        # ホバーツールにカラー軸を追加
                        hover.tooltips.append((color_by, f"@{color_by}{{0.00}}"))
                    else:
                        # 単色の散布図
                        scatter = p.circle(x=x_axis, y=y_axis, 
                                         size=6, alpha=0.6, 
                                         color='navy',
                                         source=df_sample)
                    
                    p.add_tools(hover)
                    
                    # Streamlitに表示
                    st.bokeh_chart(p, use_container_width=True)
                    
                    # 統計情報の表示
                    st.write("**統計情報**")
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        correlation = df_sample[x_axis].corr(df_sample[y_axis])
                        st.metric("相関係数", f"{correlation:.3f}")
                        
                        # 線形回帰の統計
                        slope, intercept, r_value, p_value, std_err = stats.linregress(
                            df_sample[x_axis], df_sample[y_axis]
                        )
                        st.metric("R²値", f"{r_value**2:.3f}")
                    
                    with stats_col2:
                        st.write(f"**{x_axis}統計:**")
                        st.write(f"平均: {df_sample[x_axis].mean():.2f}")
                        st.write(f"標準偏差: {df_sample[x_axis].std():.2f}")
                        
                        st.write(f"**{y_axis}統計:**")
                        st.write(f"平均: {df_sample[y_axis].mean():.2f}")
                        st.write(f"標準偏差: {df_sample[y_axis].std():.2f}")
                    
                    # 回帰直線のオプション
                    if st.checkbox("回帰直線を表示"):
                        # 回帰直線の計算
                        x_line = np.linspace(df_sample[x_axis].min(), df_sample[x_axis].max(), 100)
                        y_line = slope * x_line + intercept
                        
                        # 回帰直線をプロットに追加
                        p.line(x_line, y_line, line_width=2, color='red', alpha=0.8, legend_label='回帰直線')
                        p.legend.location = "top_left"
                        
                        # 回帰直線付きプロットを再表示
                        st.bokeh_chart(p, use_container_width=True)
                        
                        # 回帰式を表示
                        st.write(f"**回帰式:** y = {slope:.3f}x + {intercept:.3f}")
                        st.write(f"**p値:** {p_value:.3e}")
        else:
            st.warning("散布図を作成するには、少なくとも2つの数値列が必要です。")
        
        # 統計情報
        st.subheader("データ統計")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総イベント数", len(df))
        with col2:
            st.metric("パラメータ数", len(df.columns))
        with col3:
            st.metric("ファイルサイズ", f"{len(uploaded_file.getvalue())/1024/1024:.2f} MB")
        
        # メタデータ表示
        if st.checkbox("メタデータを表示"):
            st.subheader("FCSファイル メタデータ")
            meta_df = pd.DataFrame(list(meta.items()), columns=['Key', 'Value'])
            st.dataframe(meta_df)
        
        # CSVダウンロード
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="CSV をダウンロード",
            data=csv_data,
            file_name="events_output.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"FCSファイルの処理中にエラーが発生しました: {str(e)}")
        st.write("エラーの詳細:")
        st.code(str(e))
        
        # 一時ファイルが残っている場合は削除
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

else:
    st.info("FCSファイルをアップロードしてください。")
    
    # 使用方法の説明
    st.subheader("使用方法")
    st.write("""
    1. 「FCS ファイルをアップロードしてください」ボタンをクリックして、FCSファイルを選択します
    2. ファイルが正常に読み込まれると、イベントデータの上位10行が表示されます
    3. 散布図セクションで、X軸とY軸を選択して「散布図を作成」ボタンをクリックします
    4. 色分けや回帰直線の表示も可能です
    5. 「CSV をダウンロード」ボタンをクリックして、全イベントデータをCSV形式でダウンロードできます
    """)
    
    st.subheader("散布図機能")
    st.write("""
    - **軸選択**: X軸とY軸を自由に選択可能
    - **色分け**: 第3の軸で色分け表示
    - **サンプリング**: 大きなデータセットでも高速表示
    - **インタラクティブ**: ズーム、パン、ホバー機能
    - **統計情報**: 相関係数、R²値、回帰直線
    """)
    
    st.subheader("対応ファイル形式")
    st.write("- .fcs（Flow Cytometry Standard）ファイル")
    
    st.subheader("注意事項")
    st.write("""
    - 大きなFCSファイルの場合、処理に時間がかかる場合があります
    - 散布図では表示速度のため、データポイント数を制限できます
    - イベントデータは生データ（raw）として抽出されます
    - メタデータも確認できます
    """)
