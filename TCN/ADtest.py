
# 使用了一种基于支持向量回归（SVR）来预测异常分数的模型。
try:
    # For each channel in the dataset
    for channel_idx in range(nfeatures):
        ''' 1. Load mean and covariance if they are pre-calculated, if not calculate them. '''
        # Mean and covariance are calculated on train dataset.
        if 'means' in checkpoint.keys() and 'covs' in checkpoint.keys():
            print('=> loading pre-calculated mean and covariance')
            mean, cov = checkpoint['means'][channel_idx], checkpoint['covs'][channel_idx]
        else:
            print('=> calculating mean and covariance')
            mean, cov = fit_norm_distribution_param(args, model, train_dataset, channel_idx=channel_idx)

        ''' 2. Train anomaly score predictor using support vector regression (SVR). (Optional) '''
        # An anomaly score predictor is trained
        # given hidden layer output and the corresponding anomaly score on train dataset.
        # Predicted anomaly scores on test dataset can be used for the baseline of the adaptive threshold.
        if args.compensate:
            print('=> training an SVR as anomaly score predictor')
            train_score, _, _, hiddens, _ = anomalyScore(args, model, train_dataset, mean, cov, channel_idx=channel_idx)
            score_predictor = GridSearchCV(SVR(), cv=5,param_grid={"C": [1e0, 1e1, 1e2],"gamma": np.logspace(-1, 1, 3)})
            score_predictor.fit(torch.cat(hiddens,dim=0).numpy(), train_score.cpu().numpy())
        else:
            score_predictor=None

        ''' 3. Calculate anomaly scores'''
        # Anomaly scores are calculated on the test dataset
        # given the mean and the covariance calculated on the train dataset
        print('=> calculating anomaly scores')
        score, sorted_prediction, sorted_error, _, predicted_score = anomalyScore(args, model, test_dataset, mean, cov,
                                                                                  score_predictor=score_predictor,
                                                                                  channel_idx=channel_idx)

        ''' 4. Evaluate the result '''
        # The obtained anomaly scores are evaluated by measuring precision, recall, and f_beta scores
        # The precision, recall, f_beta scores are are calculated repeatedly,
        # sampling the threshold from 1 to the maximum anomaly score value, either equidistantly or logarithmically.
        print('=> calculating precision, recall, and f_beta')
        precision, recall, f_beta = get_precision_recall(args, score, num_samples=1000, beta=args.beta,
                                                         label=TimeseriesData.testLabel.to(args.device))
        print('data: ',args.data,' filename: ',args.filename,
              ' f-beta (no compensation): ', f_beta.max().item(),' beta: ',args.beta)
        if args.compensate:
            precision, recall, f_beta = get_precision_recall(args, score, num_samples=1000, beta=args.beta,
                                                             label=TimeseriesData.testLabel.to(args.device),
                                                             predicted_score=predicted_score)
            print('data: ',args.data,' filename: ',args.filename,
                  ' f-beta    (compensation): ', f_beta.max().item(),' beta: ',args.beta)


        target = preprocess_data.reconstruct(test_dataset.cpu()[:, 0, channel_idx],
                                             TimeseriesData.mean[channel_idx],
                                             TimeseriesData.std[channel_idx]).numpy()
        mean_prediction = preprocess_data.reconstruct(sorted_prediction.mean(dim=1).cpu(),
                                                      TimeseriesData.mean[channel_idx],
                                                      TimeseriesData.std[channel_idx]).numpy()
        oneStep_prediction = preprocess_data.reconstruct(sorted_prediction[:, -1].cpu(),
                                                         TimeseriesData.mean[channel_idx],
                                                         TimeseriesData.std[channel_idx]).numpy()
        Nstep_prediction = preprocess_data.reconstruct(sorted_prediction[:, 0].cpu(),
                                                       TimeseriesData.mean[channel_idx],
                                                       TimeseriesData.std[channel_idx]).numpy()
        sorted_errors_mean = sorted_error.abs().mean(dim=1).cpu()
        sorted_errors_mean *= TimeseriesData.std[channel_idx]
        sorted_errors_mean = sorted_errors_mean.numpy()
        score = score.cpu()
        scores.append(score), targets.append(target), predicted_scores.append(predicted_score)
        mean_predictions.append(mean_prediction), oneStep_predictions.append(oneStep_prediction)
        Nstep_predictions.append(Nstep_prediction)
        precisions.append(precision), recalls.append(recall), f_betas.append(f_beta)


        if args.save_fig:
            save_dir = Path('result',args.data,args.filename).with_suffix('').joinpath('fig_detection')
            save_dir.mkdir(parents=True,exist_ok=True)
            plt.plot(precision.cpu().numpy(),label='precision')
            plt.plot(recall.cpu().numpy(),label='recall')
            plt.plot(f_beta.cpu().numpy(), label='f1')
            plt.legend()
            plt.xlabel('Threshold (log scale)')
            plt.ylabel('Value')
            plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
            plt.savefig(str(save_dir.joinpath('fig_f_beta_channel'+str(channel_idx)).with_suffix('.png')))
            plt.close()


            fig, ax1 = plt.subplots(figsize=(15,5))
            ax1.plot(target,label='Target',
                     color='black',  marker='.', linestyle='--', markersize=1, linewidth=0.5)
            ax1.plot(mean_prediction, label='Mean predictions',
                     color='purple', marker='.', linestyle='--', markersize=1, linewidth=0.5)
            ax1.plot(oneStep_prediction, label='1-step predictions',
                     color='green', marker='.', linestyle='--', markersize=1, linewidth=0.5)
            ax1.plot(Nstep_prediction, label=str(args.prediction_window_size) + '-step predictions',
                     color='blue', marker='.', linestyle='--', markersize=1, linewidth=0.5)
            ax1.plot(sorted_errors_mean,label='Absolute mean prediction errors',
                     color='orange', marker='.', linestyle='--', markersize=1, linewidth=1.0)
            ax1.legend(loc='upper left')
            ax1.set_ylabel('Value',fontsize=15)
            ax1.set_xlabel('Index',fontsize=15)
            ax2 = ax1.twinx()
            ax2.plot(score.numpy().reshape(-1, 1), label='Anomaly scores from \nmultivariate normal distribution',
                     color='red', marker='.', linestyle='--', markersize=1, linewidth=1)
            if args.compensate:
                ax2.plot(predicted_score, label='Predicted anomaly scores from SVR',
                         color='cyan', marker='.', linestyle='--', markersize=1, linewidth=1)
            #ax2.plot(score.reshape(-1,1)/(predicted_score+1),label='Anomaly scores from \nmultivariate normal distribution',
            #        color='hotpink', marker='.', linestyle='--', markersize=1, linewidth=1)
            ax2.legend(loc='upper right')
            ax2.set_ylabel('anomaly score',fontsize=15)
            #plt.axvspan(2830,2900 , color='yellow', alpha=0.3)
            plt.title('Anomaly Detection on ' + args.data + ' Dataset', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.xlim([0,len(test_dataset)])
            plt.savefig(str(save_dir.joinpath('fig_scores_channel'+str(channel_idx)).with_suffix('.png')))
            #plt.show()
            plt.close()


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')