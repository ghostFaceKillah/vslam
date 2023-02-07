
"""

 MainLoop:
        switch (status_) {
            case FrontendStatus::INITING:
                BuildInitMap();
            case FrontendStatus::TRACKING_GOOD:
            case FrontendStatus::TRACKING_BAD:
                Track();
            case FrontendStatus::LOST:
                Reset();
        }



    BuildInitMap() {
      set_current_frame_as_keyframe()

      for match in matches
         tri_resu = triangulate()
         if tri_resu.good():
            add_3d_point_to_map
    }



 bool Frontend::Track() {
        num_track_last = TrackLastFrame()  // LK flow to est feature matching last <--> current frame (calcOpticalFlowPyrLK)
        // calculates feature matching against last frame (using LK optical flow)

        tracking_inliers_ = EstimateCurrentPose() // estimates current pose using g2o optimization

        if tracking_inliers_ > num_features_tracking_
            status_ = FrontendStatus::TRACKING_GOOD
        elf tracking_inliers_ > num_features_tracking_bad_
            status_ = FrontendStatus::TRACKING_BAD
        else
            status_ = FrontendStatus::LOST

        InsertKeyFrame();
    }



some further details
- prior of pose of the current frame is (last relative movement estimate) * (last frame's pose)
- g2o optimizes map points vs currently observed pose

    int Frontend::EstimateCurrentPose() {
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // vertex
        VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->Pose());
        optimizer.addVertex(vertex_pose);

        // K
        Mat33 K = camera_left_->K();

        // edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features;
        for (size_t i = 0; i < current_frame_->features_left.size(); ++i) {
            auto mp = current_frame_->features_left[i]->map_point_.lock();

            if (mp) {
                features.push_back(current_frame_->features_left[i]);
                EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->pos_, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                edge->setMeasurement(toVec2(current_frame_->features_left[i]->position_.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                index++;
            }

        }

        // estimate the Pose the determine the outliers
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration) {
            vertex_pose->setEstimate(current_frame_->Pose());
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                if (features[i]->is_outlier_) {
                    e->computeError();
                }
                if (e->chi2() > chi2_th) {
                    features[i]->is_outlier_ = true;
                    e->setLevel(1);
                    cnt_outlier++;
                } else {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);
                };

                if (iteration == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        std::cout <<  "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier << std::endl;

        // set current pose
        current_frame_->SetPose(vertex_pose->estimate());

        std::cout << "Current Pose = \n" << current_frame_->Pose().matrix() << std::endl;

        for (auto &feat : features) {
            if (feat->is_outlier_) {
                feat->map_point_.reset();
                feat->is_outlier_ = false;  // maybe we can still use it in future
            }
        }

        return features.size() - cnt_outlier;
    }


"""