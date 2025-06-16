import streamlit as st
import pandas as pd
import fcsparser
import tempfile
import os
import numpy as np
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColorBar, LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.models import ColumnDataSource

# ページ設定
st.set_page_config(
    page_title="FACS Data Analysis",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 FACS Data Analysis")
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
        
        st.success(f"✅ FCSファイルが正常に読み込まれました。{len(df):,} イベントが検出されました。")
        
        # データのプレビュー表示
        st.subheader("📋 データプレビュー（上位10行）")
        st.dataframe(df.head(10))
        
        # 数値列のみを取得
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 蛍光強度統計情報セクション
        st.subheader("📊 蛍光強度統計情報")
        if len(numeric_columns) > 0:
            # 統計情報を計算
            stats_summary = df[numeric_columns].agg(['mean', 'std', 'min', 'max', 'count'])
            stats_summary = stats_summary.round(3)
            
            # 統計表を表示
            st.write("### 📈 各パラメータの統計サマリー")
            st.dataframe(stats_summary.T, use_container_width=True)
            
            # 特定のパラメータの詳細統計
            st.write("### 🎯 パラメータ別詳細統計")
            selected_param = st.selectbox("詳細を見るパラメータを選択:", numeric_columns)
            
            if selected_param:
                param_stats = df[selected_param].describe()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("平均値", f"{param_stats['mean']:.3f}")
                with col2:
                    st.metric("標準偏差", f"{param_stats['std']:.3f}")
                with col3:
                    st.metric("最小値", f"{param_stats['min']:.3f}")
                with col4:
                    st.metric("最大値", f"{param_stats['max']:.3f}")
        
        # ヒストグラム作成セクション
        st.subheader("📊 ヒストグラム作成")
        if len(numeric_columns) > 0:
            hist_col1, hist_col2 = st.columns(2)
            
            with hist_col1:
                hist_param = st.selectbox("ヒストグラムを作成するパラメータ:", numeric_columns, key="hist_param")
            
            with hist_col2:
                bin_count = st.slider("ビン数", min_value=10, max_value=100, value=50, step=5)
            
            # ヒストグラムオプション
            hist_options_col1, hist_options_col2 = st.columns(2)
            with hist_options_col1:
                hist_width = st.slider("ヒストグラム幅", 400, 1000, 700, key="hist_width")
            with hist_options_col2:
                hist_height = st.slider("ヒストグラム高さ", 300, 600, 400, key="hist_height")
            
            if st.button("📊 ヒストグラムを作成", type="primary", key="create_histogram"):
                with st.spinner('ヒストグラムを作成中...'):
                    # データを取得
                    hist_data = df[hist_param].dropna()
                    
                    # ヒストグラムデータを計算
                    hist, edges = np.histogram(hist_data, bins=bin_count)
                    
                    # Bokehでヒストグラムを作成
                    p_hist = figure(width=hist_width, height=hist_height,
                                   title=f"{hist_param} ヒストグラム (n={len(hist_data):,})",
                                   x_axis_label=hist_param,
                                   y_axis_label="頻度",
                                   tools="pan,wheel_zoom,box_zoom,reset,save")
                    
                    # バーを描画
                    p_hist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                               fill_color="navy", line_color="white", alpha=0.7)
                    
                    # グリッドスタイルの調整
                    p_hist.grid.grid_line_alpha = 0.3
                    
                    # 統計線を追加
                    mean_val = hist_data.mean()
                    std_val = hist_data.std()
                    
                    # 平均線
                    p_hist.line([mean_val, mean_val], [0, max(hist)], 
                               line_color="red", line_width=2, legend_label=f"平均: {mean_val:.3f}")
                    
                    # ±1標準偏差線
                    p_hist.line([mean_val - std_val, mean_val - std_val], [0, max(hist) * 0.8], 
                               line_color="orange", line_width=2, line_dash="dashed", 
                               legend_label=f"-1σ: {mean_val - std_val:.3f}")
                    p_hist.line([mean_val + std_val, mean_val + std_val], [0, max(hist) * 0.8], 
                               line_color="orange", line_width=2, line_dash="dashed", 
                               legend_label=f"+1σ: {mean_val + std_val:.3f}")
                    
                    # 凡例の設定
                    p_hist.legend.location = "top_right"
                    p_hist.legend.click_policy = "hide"
                    
                    # Streamlitに表示
                    st.bokeh_chart(p_hist, use_container_width=True)
                    
                    # ヒストグラム統計情報
                    st.info(f"""
                    📊 **ヒストグラム統計情報**
                    - パラメータ: {hist_param}
                    - データ数: {len(hist_data):,}
                    - ビン数: {bin_count}
                    - 平均値: {mean_val:.3f}
                    - 標準偏差: {std_val:.3f}
                    - 範囲: {hist_data.min():.3f} ～ {hist_data.max():.3f}
                    """)
        
        # 散布図作成セクション（既存コード）
        st.subheader("📊 散布図作成")
        
        if len(numeric_columns) >= 2:
            # 軸選択
            st.write("### 🎯 軸選択")
            col1, col2 = st.columns(2)
            
            with col1:
                x_axis = st.selectbox("X軸を選択してください", numeric_columns, key="x_axis")
            
            with col2:
                y_axis = st.selectbox("Y軸を選択してください", 
                                    [col for col in numeric_columns if col != x_axis], 
                                    key="y_axis")
            
            # プロット範囲設定
            st.write("### 🎛️ プロット範囲設定")
            
            # データの範囲を取得
            x_min, x_max = float(df[x_axis].min()), float(df[x_axis].max())
            y_min, y_max = float(df[y_axis].min()), float(df[y_axis].max())
            
            range_col1, range_col2 = st.columns(2)
            
            with range_col1:
                st.write(f"**🔢 X軸範囲設定 ({x_axis})**")
                st.write(f"データ範囲: {x_min:.2f} ～ {x_max:.2f}")
                
                use_custom_x = st.checkbox("カスタムX軸範囲を使用", key="custom_x_range")
                
                if use_custom_x:
                    x_range_method = st.radio(
                        "X軸設定方法:",
                        ["スライダー", "数値入力"],
                        horizontal=True,
                        key="x_range_method"
                    )
                    
                    if x_range_method == "スライダー":
                        x_range = st.slider(
                            "X軸表示範囲",
                            min_value=x_min,
                            max_value=x_max,
                            value=(x_min, x_max),
                            step=(x_max - x_min) / 100,
                            key="x_range_slider",
                            format="%.2f"
                        )
                    else:
                        x_input_col1, x_input_col2 = st.columns(2)
                        with x_input_col1:
                            x_min_input = st.number_input(
                                "X軸最小値",
                                value=x_min,
                                step=(x_max - x_min) / 100,
                                format="%.4f",
                                key="x_min_input"
                            )
                        with x_input_col2:
                            x_max_input = st.number_input(
                                "X軸最大値",
                                value=x_max,
                                step=(x_max - x_min) / 100,
                                format="%.4f",
                                key="x_max_input"
                            )
                        x_range = (x_min_input, x_max_input)
                else:
                    x_range = (x_min, x_max)
            
            with range_col2:
                st.write(f"**📊 Y軸範囲設定 ({y_axis})**")
                st.write(f"データ範囲: {y_min:.2f} ～ {y_max:.2f}")
                
                use_custom_y = st.checkbox("カスタムY軸範囲を使用", key="custom_y_range")
                
                if use_custom_y:
                    y_range_method = st.radio(
                        "Y軸設定方法:",
                        ["スライダー", "数値入力"],
                        horizontal=True,
                        key="y_range_method"
                    )
                    
                    if y_range_method == "スライダー":
                        y_range = st.slider(
                            "Y軸表示範囲",
                            min_value=y_min,
                            max_value=y_max,
                            value=(y_min, y_max),
                            step=(y_max - y_min) / 100,
                            key="y_range_slider",
                            format="%.2f"
                        )
                    else:
                        y_input_col1, y_input_col2 = st.columns(2)
                        with y_input_col1:
                            y_min_input = st.number_input(
                                "Y軸最小値",
                                value=y_min,
                                step=(y_max - y_min) / 100,
                                format="%.4f",
                                key="y_min_input"
                            )
                        with y_input_col2:
                            y_max_input = st.number_input(
                                "Y軸最大値",
                                value=y_max,
                                step=(y_max - y_min) / 100,
                                format="%.4f",
                                key="y_max_input"
                            )
                        y_range = (y_min_input, y_max_input)
                else:
                    y_range = (y_min, y_max)
            
            # プロットオプション
            st.write("### 🎨 プロットオプション")
            plot_col1, plot_col2, plot_col3 = st.columns(3)
            
            with plot_col1:
                sample_size = st.slider("表示するデータポイント数", 
                                      min_value=1000, 
                                      max_value=min(len(df), 100000), 
                                      value=min(10000, len(df)),
                                      step=1000)
            
            with plot_col2:
                color_by = st.selectbox("色分けする軸（オプション）", 
                                      ["なし"] + numeric_columns, 
                                      key="color_axis")
            
            with plot_col3:
                alpha_value = st.slider("透明度", 0.1, 1.0, 0.6, 0.1)
            
            # プロットサイズ設定
            size_col1, size_col2 = st.columns(2)
            with size_col1:
                plot_width = st.slider("プロット幅", 400, 1000, 700)
            with size_col2:
                plot_height = st.slider("プロット高さ", 300, 800, 500)
            
            # 範囲内データの統計表示
            st.write("### 📈 選択範囲内のデータ統計")
            
            # 範囲内のデータをフィルタ
            filtered_df = df[
                (df[x_axis] >= x_range[0]) & (df[x_axis] <= x_range[1]) &
                (df[y_axis] >= y_range[0]) & (df[y_axis] <= y_range[1])
            ]
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("選択範囲内イベント数", f"{len(filtered_df):,}")
            with stats_col2:
                st.metric("全体に対する割合", f"{len(filtered_df)/len(df)*100:.1f}%")
            with stats_col3:
                if len(filtered_df) > 0:
                    st.metric(f"{x_axis} 平均", f"{filtered_df[x_axis].mean():.2f}")
                else:
                    st.metric(f"{x_axis} 平均", "N/A")
            with stats_col4:
                if len(filtered_df) > 0:
                    st.metric(f"{y_axis} 平均", f"{filtered_df[y_axis].mean():.2f}")
                else:
                    st.metric(f"{y_axis} 平均", "N/A")
            
            # プロット作成ボタン
            if st.button("🚀 散布図を作成", type="primary", use_container_width=True):
                if len(filtered_df) == 0:
                    st.warning("⚠️ 選択した範囲内にデータがありません。範囲を調整してください。")
                else:
                    with st.spinner('散布図を作成中...'):
                        # データをサンプリング（範囲内データから）
                        if len(filtered_df) > sample_size:
                            df_sample = filtered_df.sample(n=sample_size, random_state=42)
                        else:
                            df_sample = filtered_df.copy()
                        
                        # ColumnDataSourceを作成
                        source = ColumnDataSource(df_sample)
                        
                        # Bokeh散布図の作成
                        p = figure(width=plot_width, height=plot_height,
                                 title=f"{y_axis} vs {x_axis} (選択範囲内: {len(df_sample):,} points)",
                                 x_axis_label=x_axis,
                                 y_axis_label=y_axis,
                                 tools="pan,wheel_zoom,box_zoom,reset,save,lasso_select,box_select")
                        
                        # 軸範囲を設定
                        p.x_range.start = x_range[0]
                        p.x_range.end = x_range[1]
                        p.y_range.start = y_range[0]
                        p.y_range.end = y_range[1]
                        
                        # ホバーツールの設定
                        hover_tooltips = [
                            ("Index", "$index"),
                            (x_axis, f"@{{{x_axis}}}{{0.00}}"),
                            (y_axis, f"@{{{y_axis}}}{{0.00}}")
                        ]
                        
                        if color_by != "なし":
                            # カラーマッピングの設定（フィルタ済みデータの範囲で）
                            color_min = df_sample[color_by].min()
                            color_max = df_sample[color_by].max()
                            
                            if color_min != color_max:  # 単一値でない場合のみカラーマッピング
                                color_mapper = LinearColorMapper(palette=Viridis256, 
                                                               low=color_min, 
                                                               high=color_max)
                                
                                # カラーバー付きの散布図
                                scatter = p.circle(x=x_axis, y=y_axis, 
                                                 size=6, alpha=alpha_value,
                                                 color={'field': color_by, 'transform': color_mapper},
                                                 source=source)
                                
                                # カラーバーを追加
                                color_bar = ColorBar(color_mapper=color_mapper, 
                                                   label_standoff=12,
                                                   location=(0,0),
                                                   title=color_by)
                                p.add_layout(color_bar, 'right')
                                
                                # ホバーツールにカラー軸を追加
                                hover_tooltips.append((color_by, f"@{{{color_by}}}{{0.00}}"))
                            else:
                                # 単色の散布図（単一値の場合）
                                scatter = p.circle(x=x_axis, y=y_axis, 
                                                 size=6, alpha=alpha_value, 
                                                 color='navy',
                                                 source=source)
                                st.info(f"色分け軸 '{color_by}' の値が単一のため、色分けは適用されません。")
                        else:
                            # 単色の散布図
                            scatter = p.circle(x=x_axis, y=y_axis, 
                                             size=6, alpha=alpha_value, 
                                             color='navy',
                                             source=source)
                        
                        # ホバーツールを追加
                        hover = HoverTool(tooltips=hover_tooltips)
                        p.add_tools(hover)
                        
                        # グリッドスタイルの調整
                        p.grid.grid_line_alpha = 0.3
                        
                        # Streamlitに表示
                        st.bokeh_chart(p, use_container_width=True)
                        
                        # プロット情報
                        st.info(f"""
                        📊 **プロット情報**
                        - 表示データポイント数: {len(df_sample):,}
                        - X軸範囲: {x_range[0]:.3f} ～ {x_range[1]:.3f}
                        - Y軸範囲: {y_range[0]:.3f} ～ {y_range[1]:.3f}
                        - 色分け: {color_by if color_by != "なし" else "単色"}
                        """)
        else:
            st.warning("⚠️ 散布図を作成するには、少なくとも2つの数値列が必要です。")
        
        # 統計情報
        st.subheader("📊 データ統計")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総イベント数", f"{len(df):,}")
        with col2:
            st.metric("パラメータ数", len(df.columns))
        with col3:
            st.metric("ファイルサイズ", f"{len(uploaded_file.getvalue())/1024/1024:.2f} MB")
        
        # 数値パラメータの統計サマリー
        if len(numeric_columns) > 0:
            st.subheader("📈 数値パラメータ統計サマリー")
            stats_df = df[numeric_columns].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        # メタデータ表示
        if st.checkbox("🔍 メタデータを表示"):
            st.subheader("📋 FCSファイル メタデータ")
            meta_df = pd.DataFrame(list(meta.items()), columns=['Key', 'Value'])
            st.dataframe(meta_df, use_container_width=True)
        
        # データダウンロード
        st.subheader("💾 データダウンロード")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            # 全データのダウンロード
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📄 全データをCSVでダウンロード",
                data=csv_data,
                file_name="facs_all_events.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with download_col2:
            # 選択範囲内データのダウンロード（軸が選択されている場合）
            if 'x_axis' in st.session_state and 'y_axis' in st.session_state:
                if 'filtered_df' in locals() and len(filtered_df) > 0:
                    filtered_csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="🎯 選択範囲内データをCSVでダウンロード",
                        data=filtered_csv,
                        file_name=f"facs_filtered_{x_axis}_{y_axis}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.button("🎯 選択範囲内データをCSVでダウンロード", 
                             disabled=True, 
                             help="選択範囲内にデータがありません",
                             use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ FCSファイルの処理中にエラーが発生しました: {str(e)}")
        with st.expander("エラーの詳細"):
            st.code(str(e))
        
        # 一時ファイルが残っている場合は削除
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

else:
    st.info("📁 FCSファイルをアップロードしてください。")
    
    # 使用方法の説明
    st.subheader("📖 使用方法")
    with st.container():
        st.markdown("""
        ### 基本的な流れ
        1. **ファイルアップロード**: 「FCS ファイルをアップロードしてください」ボタンをクリックして、FCSファイルを選択
        2. **データ確認**: ファイルが正常に読み込まれると、イベントデータの上位10行が表示
        3. **統計情報確認**: 各蛍光強度パラメータの平均値・標準偏差を確認
        4. **ヒストグラム作成**: 特定のパラメータのヒストグラムを作成（平均・標準偏差線付き）
        5. **軸選択**: 散布図セクションで、X軸とY軸を選択
        6. **範囲設定**: プロット範囲設定で表示する範囲を細かく調整
        7. **プロット作成**: 「散布図を作成」ボタンをクリックして可視化
        8. **データダウンロード**: 必要に応じて全データまたは選択範囲内データをCSVでダウンロード
        """)
    
    st.subheader("📊 新機能：統計情報・ヒストグラム")
    with st.container():
        st.markdown("""
        ### 蛍光強度統計情報
        - **全パラメータ統計**: 各蛍光強度の平均値・標準偏差・最小値・最大値を一覧表示
        - **詳細統計**: 特定パラメータの詳細統計情報をメトリック表示
        
        ### ヒストグラム機能
        - **パラメータ選択**: 任意の蛍光強度パラメータでヒストグラム作成
        - **ビン数調整**: 10〜100の範囲でヒン数を調整可能
        - **統計線表示**: 平均値線と±1標準偏差線を自動表示
        - **サイズ調整**: ヒストグラムの幅・高さを自由に調整
        - **詳細情報**: データ数、統計値、範囲を表示
        """)
    
    st.subheader("🎛️ プロット範囲設定機能")
    with st.container():
        st.markdown("""
        ### 詳細範囲設定
        - **スライダー設定**: 直感的なスライダーで範囲を調整
        - **数値入力設定**: 正確な数値で範囲を指定
        - **リアルタイム統計**: 選択範囲内のデータ統計を即座に表示
        - **範囲内データ抽出**: 設定した範囲内のデータのみを可視化・ダウンロード
        - **自動範囲調整**: プロット表示範囲が自動的に設定範囲に調整
        """)
    
    st.subheader("📊 散布図機能")
    with st.container():
        st.markdown("""
        - **軸選択**: X軸とY軸を自由に選択可能
        - **色分け**: 第3の軸で色分け表示（カラーバー付き）
        - **サンプリング**: 大きなデータセットでも高速表示
        - **インタラクティブ**: ズーム、パン、ホバー、選択機能
        - **透明度調整**: データの重なりを見やすく調整
        - **カスタムサイズ**: プロットサイズを自由に調整
        """)
    
    st.subheader("📁 対応ファイル形式")
    st.write("- .fcs（Flow Cytometry Standard）ファイル")
    
    st.subheader("⚠️ 注意事項")
    with st.container():
        st.markdown("""
        - 大きなFCSファイルの場合、処理に時間がかかる場合があります
        - 散布図では表示速度のため、データポイント数を制限できます
        - イベントデータは生データ（raw）として抽出されます
        - メタデータも確認できます
        - 選択範囲外のデータは表示されません
        - ヒストグラムでは欠損値（NA）は自動的に除外されます
        """)
