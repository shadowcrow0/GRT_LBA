# 修正後重新執行 PPC 模擬與儲存流程（使用 GRT participant ID 為主）

all_ppc_dfs = []

for grt_pid in params_df['participant_id']:
    param_row = params_df[params_df['participant_id'] == grt_pid]
    if param_row.empty:
        continue
    param_row = param_row.iloc[0].to_dict()

    df = raw_df[raw_df['participant'] == grt_pid].copy()
    df = df.rename(columns={'stim_condition': 'stimulus_condition'})
    df['trial_id'] = df.index

    predicted_rts = []
    used_left_drift = []
    used_right_drift = []

    for _, row in df.iterrows():
        l_stim, r_stim, resp = row['Chanel1'], row['Chanel2'], int(row['Response'])

        # 左側 drift rate
        if l_stim == 1:
            l_v = param_row['left_v_vertical']
            l_v_err = param_row['left_v_vertical_error']
        else:
            l_v = param_row['left_v_nonvertical']
            l_v_err = param_row['left_v_nonvertical_error']

        # 右側 drift rate
        if r_stim == 1:
            r_v = param_row['right_v_vertical']
            r_v_err = param_row['right_v_vertical_error']
        else:
            r_v = param_row['right_v_nonvertical']
            r_v_err = param_row['right_v_nonvertical_error']

        left_drift = l_v if resp in [1, 2] else l_v_err
        right_drift = r_v if resp in [0, 1] else r_v_err
        used_left_drift.append(left_drift)
        used_right_drift.append(right_drift)

        eff_drift = max(min(left_drift, right_drift), 0.05)
        pred_rt = param_row['threshold'] / eff_drift + param_row['ndt']
        predicted_rts.append(pred_rt)

    df['predicted_rt'] = predicted_rts
    df['v_left'] = used_left_drift
    df['v_right'] = used_right_drift

    df.to_csv(output_dir / f"participant_{grt_pid}_ppc.csv", index=False)
    all_ppc_dfs.append(df[['participant', 'trial_id', 'RT', 'predicted_rt']])

    # 畫 CDF 圖
    obs_rt = df['RT'].values
    pred_rt = df['predicted_rt'].values
    bins = np.linspace(0, max(obs_rt.max(), pred_rt.max()), 50)

    plt.figure(figsize=(6, 4))
    plt.hist(obs_rt, bins=bins, density=True, histtype='step', label='Observed RT', cumulative=True)
    plt.hist(pred_rt, bins=bins, density=True, histtype='step', label='Predicted RT', cumulative=True)
    plt.title(f"Participant {grt_pid} - RT CDF")
    plt.xlabel("RT (s)")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"participant_{grt_pid}_cdf.png")
    plt.close()

# 儲存總結資料
combined_ppc_df = pd.concat(all_ppc_dfs, ignore_index=True)
combined_ppc_df.to_csv(output_dir / "all_participants_ppc_summary.csv", index=False)

output_dir
