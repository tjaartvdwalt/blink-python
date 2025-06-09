
def main():



if __name__ == "__main__":
    main()

cv::CascadeClassifier BlinkData::faces_cascade =
    BlinkData::load_cascade(cv::samples::findFile(
        "haarcascades/haarcascade_frontalface_alt.xml", false, true));
cv::CascadeClassifier BlinkData::eyes_cascade =
    BlinkData::load_cascade(cv::samples::findFile(
        "haarcascades/haarcascade_eye_tree_eyeglasses.xml", false, true));

int main(int argc, const char** argv)
{
    cv::CommandLineParser parser = parse_args(argc, argv, keys);
    load_cascades(parser);


    std::string input_video = parser.get<std::string>(0);
    std::string output_video = parser.get<std::string>(1);
    int blinks_left = 0;
    int blinks_right = 0;
    int blinks_full = 0;

    if (output_video.length() == 0)
    {
        output_video = "output.mp4v";
    }

    BlinkAnnotations* annotations = new BlinkAnnotations("file.tag");

    try
    {
        cv::VideoCapture video;
        video.open(input_video);

        if (!video.isOpened())
        {
            throw(std::string("Unable to open video file ") + input_video);
        }

        // video properties
        double width = video.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = video.get(cv::CAP_PROP_FPS);

        int max_height = parser.get<int>("max_height");

        double scale = 1.0;
        if (height > max_height)
        {
            scale = max_height / height;
        }

        cv::Mat frame;
        std::string orig("Original video");
        cv::namedWindow(orig);

        int max_size = 13;
        FrameWindow window = FrameWindow(max_size);

        cv::VideoWriter out(output_video,
            cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), fps,
            cv::Size(width * scale, height * scale));

        Detector* detector;
        std::string detector_type = parser.get<std::string>("detector");
        if (detector_type == "area")
        {
            detector = new AreaDetector(max_size);
        }
        else if (detector_type == "iris")
        {
            detector = new IrisDetector();
        }
        else if (detector_type == "landmark")
        {
            detector = new LandmarkDetector();
        }
        else
        {
            detector = new SimpleDetector();
        }

        SimpleBlinkJudge judge = SimpleBlinkJudge();

        while (video.isOpened())
        {
            video >> frame;

            if (frame.empty())
                break;

            cv::resize(frame, frame, cv::Size(width * scale, height * scale));
            int frame_number = video.get(cv::CAP_PROP_POS_FRAMES);

            if (frame_number < parser.get<int>("start_frame"))
            {
                continue;
            }
            cv::Mat gray_frame = preprocess_frame(frame);

            window.add(new FrameData(frame_number, &frame,
                new BlinkData(&gray_frame), new BlinkMeta(frame_number, fps)));

            if (window.isFull())
            {
                FrameData* cur = window.getCur();

                bool success = detector->detect(&window);
                if (!success)
                {
                    continue;
                }

                judge.judgeBlink(&window);

                if (cur->meta()->blinkStateLeft() == BlinkState::start)
                {
                    cur->meta()->blinksLeft(++blinks_left);
                }
                else
                {
                    cur->meta()->blinksLeft(blinks_left);
                }
                if (cur->meta()->blinkStateRight() == BlinkState::start)
                {
                    cur->meta()->blinksRight(++blinks_right);
                }
                else
                {
                    cur->meta()->blinksRight(blinks_right);
                }
                if (cur->meta()->blinkStateFull() == BlinkState::start)
                {
                    cur->meta()->blinksFull(++blinks_full);
                }
                else
                {
                    cur->meta()->blinksFull(blinks_full);
                }
#ifdef _DEBUG
                print_debug(cur);
#endif
                annotations->annotateFrame(cur);

                annotate_video_frame(cur, frame);
                out.write(frame);

                cv::imshow(orig, frame);
                if (cv::waitKey(1) == 'q')
                {
                    break;
                }
            }
            else
            {
                out.write(frame);
            }
        }
        video.release();
        out.release();
    }
    catch (std::runtime_error& exc)
    {
        std::cerr << "Error: " << exc.what() << std::endl;
        return (1);
    }
    catch (cv::Exception& exc)
    {
        std::cerr << "Error: " << exc.msg << std::endl;
        return (1);
    }

    return (0);
}






class LandmarkDetector : public Detector
{
  private:
    /* cv::Ptr<cv::face::Facemark> facemark; */
    dlib::frontal_face_detector detector;
    dlib::shape_predictor landmark_model;
    cv::Ptr<cv::ml::SVM> svm;
    cv::dnn::Net net;
    int inter_eye_distance = 0;

    float dist(dlib::point p1, dlib::point p2)
    {
        return sqrt(pow(p2.x() - p1.x(), 2) + pow(p2.y() - p1.y(), 2));
    }

    std::array<float, 2> movingAverageOfAspectRatios(FrameWindow* window)
    {
        float sum_left_aspect_ratio = 0;
        float sum_right_aspect_ratio = 0;
        int left_count = 0;
        int right_count = 0;

        for (int i = 0; i < window->getSize(); i++)
        {
            FrameData* frame = window->get(i);
            sum_left_aspect_ratio += frame->meta()->aspectRatioLeft();
            // std::cout << frame->meta()->aspectRatioLeft() << ", ";
            left_count++;
            sum_right_aspect_ratio += frame->meta()->aspectRatioRight();
            right_count++;
        }

        float left_aspect_ratio = sum_left_aspect_ratio / left_count;
        float right_aspect_ratio = sum_right_aspect_ratio / right_count;

        std::array<float, 2> returnArray = {
            left_aspect_ratio, right_aspect_ratio};
        return returnArray;
    }

    int interEyeDistance(dlib::full_object_detection landmarks)
    {
        cv::Point2f leftEyeLeftCorner =
            cv::Point2f(landmarks.part(36).x(), landmarks.part(36).y());
        cv::Point2f rightEyeRightCorner =
            cv::Point2f(landmarks.part(45).x(), landmarks.part(45).y());
        double dist = cv::norm(rightEyeRightCorner - leftEyeLeftCorner);
        return (int)dist;
    }

    void stabilizePoints(
        dlib::full_object_detection landmarks, FrameData* prev, FrameData* cur)
    {
        cv::Mat prev_frame;
        cv::Mat cur_frame;
        cv::cvtColor(*prev->frame(), prev_frame, cv::COLOR_BGR2GRAY);
        cv::cvtColor(*cur->frame(), cur_frame, cv::COLOR_BGR2GRAY);
        if (this->inter_eye_distance == 0)
        {
            this->inter_eye_distance = interEyeDistance(landmarks);
        }

        float sigma = this->inter_eye_distance * this->inter_eye_distance / 400;
        float s = 2 * int(this->inter_eye_distance / 4) + 1;
        std::vector<uchar> status;
        std::vector<float> err;
        int maxLevel = 5;
        cv::Size winSize(s, s);
        cv::TermCriteria termcrit(
            cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
        // int flags = 0;
        // double minEigThreshold = 0.001;
        std::vector<cv::Point2f> points;

        cv::calcOpticalFlowPyrLK(prev_frame, cur_frame, *prev->points(), points,
            status, err, winSize, maxLevel, termcrit);
        for (long unsigned int i = 0; i < cur->pointsDetected()->size(); i++)
        {
            double d = cv::norm(
                prev->pointsDetected()->at(i) - cur->pointsDetected()->at(i));
            double alpha = exp(-d * d / sigma);
            // double point_detected_arr[] = {cur->pointsDetected().at(i).x(),
            // cur->pointsDetected().at(i).y()}; double point_arr[] =
            // {points.at(i).x(), points.at(i).y()};

            double val_x = (1 - alpha) * cur->pointsDetected()->at(i).x +
                           alpha * points.at(i).x;
            double val_y = (1 - alpha) * cur->pointsDetected()->at(i).y +
                           alpha * points.at(i).y;
            // double val_y
            // (1 - alpha) * points.at(i).y + alpha * cur->points()->at(i).y;
            cur->points()->at(i) = cv::Point2f(val_x, val_y);
        }
    }

    void calculateAspectRatios(FrameData* cur, FrameData* prev)
    {
        if (!cur || !cur->data()->getFace())
        {
            return;
        }

        cv::Mat img = *cur->frame();
        /* cv::Mat resized; */
        /* resize(*cur->frame(), resized, cv::Size(img.cols * 2,
         * img.rows * 2), cv::INTER_LINEAR); */

        dlib::cv_image<dlib::bgr_pixel> cimg(img);
        std::vector<dlib::rectangle> faces = this->detector(cimg);

        if (faces.size() > 0)
        {
            dlib::full_object_detection shape =
                this->landmark_model(cimg, faces[0]);
            cur->pointsDetected(shape);

            if (prev == NULL)
            {
                for (long unsigned int i = 0; i < cur->pointsDetected()->size();
                     i++)
                {
                    cur->points()->at(i) = cur->pointsDetected()->at(i);
                }
            }
            else
            {
                this->stabilizePoints(shape, prev, cur);
            }

            if (cur->points()->size() > 0)
            {
                float right_vert_1 = (float)cv::norm(
                    cur->points()->at(37) - cur->points()->at(41));
                float right_vert_2 = (float)cv::norm(
                    cur->points()->at(38) - cur->points()->at(40));
                float left_vert_1 = (float)cv::norm(
                    cur->points()->at(43) - cur->points()->at(47));
                float left_vert_2 = (float)cv::norm(
                    cur->points()->at(44) - cur->points()->at(46));

                float left_horz = (float)cv::norm(
                    cur->points()->at(36) - cur->points()->at(39));
                float right_horz = (float)cv::norm(
                    cur->points()->at(42) - cur->points()->at(45));

                // float right_vert_1 = this->dist(shape.part(37),
                // shape.part(41)); float right_vert_2 =
                // this->dist(shape.part(38), shape.part(40)); float left_vert_1
                // = this->dist(shape.part(43), shape.part(47)); float
                // left_vert_2 = this->dist(shape.part(44), shape.part(46));
                //
                // float left_horz = dist(shape.part(36), shape.part(39));
                // float right_horz = dist(shape.part(42), shape.part(45));

                // for (int j : {36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47})
                // {
                //     cv::circle(*cur->frame(),
                //         // *cur->points()->at(j),
                //         cv::Point((int)cur->points()->at(j).x,
                //             (int)cur->points()->at(j).y),
                //         2, cv::Scalar(0, 0, 255), -1);
                // }

                double left_aspect_ratio =
                    (left_vert_1 + left_vert_2) / (2 * left_horz);
                double right_aspect_ratio =
                    (right_vert_1 + right_vert_2) / (2 * right_horz);

                cur->meta()->aspectRatioLeft(left_aspect_ratio);
                cur->meta()->aspectRatioRight(right_aspect_ratio);
            }
        }
        /* std::vector<cv::Rect> faces; */
        /* faces.push_back(*cur->data()->getFace()); */
        /* std::vector<std::vector<cv::Point2f>> landmarks; */

        /* if (this->facemark->fit(img, faces, landmarks)) */
        /* { */
        /*     std::vector<cv::Point_<float>> first = landmarks.at(0);
         */
        /*     float right_vert_1 = this->dist(first.at(37),
         * first.at(41)); */
        /*     float right_vert_2 = this->dist(first.at(38),
         * first.at(40)); */
        /*     float left_vert_1 = this->dist(first.at(43),
         * first.at(47)); */
        /*     float left_vert_2 = this->dist(first.at(44),
         * first.at(46)); */

        /* float aspect_ratio = (left_aspect_ratio + right_aspect_ratio)
         * / 2; */

        /* std::array<float, 2> avg_aspect_ratios = */
        /*     this->movingAverageOfAspectRatios(window); */

        /* float data[1][13] =
         * {{1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, */
        /* 1.1, 1.1, 1.1}}; */
        /* for (int i = 0; i < 13; i++) */
        /* { */
        /*     pred.at<float>(0, i) = 1.1; */
        /* } */

        /* cv::Mat pred(1, 13, CV_32F, data); */
        /* float avg_left_aspect_ratio = avg_aspect_ratios[0]; */
        /* float avg_right_aspect_ratio = avg_aspect_ratios[1]; */

        /* empty->meta()->aspectRatioRight(avg_right_aspect_ratio); */
        /* } */
    }

    void plotEAR(FrameWindow* window)
    {
        cv::Mat data_x(1, window->getSize(), CV_64F);
        cv::Mat data_y(1, window->getSize(), CV_64F);

        for (int i = 0; i < window->getSize(); i++)
        {
            data_x.at<double>(0, i) = window->get(i)->frameNumber();
            data_y.at<double>(0, i) = window->get(i)->meta()->aspectRatioLeft();
        }

        cv::Mat plot_result;

        cv::Ptr<cv::plot::Plot2d> plot =
            cv::plot::Plot2d::create(data_x, data_y);
        plot->render(plot_result);

        /* imshow("The plot rendered with default visualization options", */
        /*     plot_result); */

        /* plot->setShowText(false); */
        /* plot->setShowGrid(false); */
        /* plot->setPlotBackgroundColor(cv::Scalar(255, 200, 200)); */
        /* plot->setPlotLineColor(cv::Scalar(255, 0, 0)); */
        /* plot->setPlotLineWidth(2); */
        /* plot->setInvertOrientation(true); */
        /* plot->render(plot_result); */
        /**/
        /* imshow("The plot rendered with some of custom visualization
         * options",
         */
        /*     plot_result); */
        /* cv::waitKey(); */
    }

  public:
    LandmarkDetector()
    {
        /* this->facemark = cv::face::createFacemarkLBF(); */
        /* this->facemark->loadModel("/home/tjaart/Code/github.com/tjaartvdwalt/"
         */
        /* "blink/models/lbfmodel.yaml"); */
        this->detector = dlib::get_frontal_face_detector();
        dlib::deserialize(
            "/home/tjaart/Code/github.com/tjaartvdwalt/blink/models/"
            "shape_predictor_68_face_landmarks.dat") >>
            this->landmark_model;
        /* this->svm = cv::ml::SVM::create(); */
        this->svm =
            cv::ml::SVM::load("/home/tjaart/Code/github.com/tjaartvdwalt/"
                              "blink/models/svm.xml");
        this->net = cv::dnn::readNet("/home/tjaart/Code/github.com/"
                                     "tjaartvdwalt/blink/models/blink.onnx");
        /* this->sum_left_aspect_ratio = 0; */
        /* this->count_left = 0; */
        /* this->count_right = 0; */
        /* this->sum_right_aspect_ratio = 0; */
    }
    // Simple eye detector... if we detect an eye region, the eye is open,
    // if we don't, it is closed
    bool detect(FrameWindow* window)
    {
        for (int i = 0; i < window->getSize(); i++)
        {
            if (!window->get(i)->meta()->aspectRatioLeft() ||
                !window->get(i)->meta()->aspectRatioRight())
            {
                FrameData* prev = NULL;
                if (i > 0)
                {
                    prev = window->get(i - 1);
                }
                FrameData* cur = window->get(i);

                this->calculateAspectRatios(cur, prev);
            }
        }

        /* plotEAR(window); */

        cv::Mat pred_left(1, window->getSize(), CV_32F);
        cv::Mat pred_right(1, window->getSize(), CV_32F);

        for (int i = 0; i < window->getSize(); i++)
        {
            FrameData* frame = window->get(i);

            pred_left.at<float>(0, i) = frame->meta()->aspectRatioLeft();
            pred_right.at<float>(0, i) = frame->meta()->aspectRatioRight();
        }


        this->net.setInput(pred_left);

        // Forward propagate.
        std::vector<cv::Mat> outputs_l;
        net.forward(outputs_l, net.getUnconnectedOutLayersNames());
        cv::Mat mat_l = outputs_l.at(0);

        // std::cout << outputs_l.at(0, 0) << std::endl;

        bool blink_left = false;
        if (mat_l.at<float>(0) < mat_l.at<float>(1))
        {
            blink_left = true;
        }

        this->net.setInput(pred_right);

        // Forward propagate.
        std::vector<cv::Mat> outputs_r;
        net.forward(outputs_r, net.getUnconnectedOutLayersNames());
        cv::Mat mat_r = outputs_r.at(0);

        // std::cout << mat_r.type() << std::endl;
        // std::cout << mat_r.at<float>(0) << std::endl;
        // std::cout << mat_r.at<float>(1) << std::endl;

        bool blink_right = false;
        if (mat_r.at<float>(0) < mat_r.at<float>(1))
        {
            blink_right = true;
        }

        // float response_left = this->svm->predict(pred_left);
        // float response_right = this->svm->predict(pred_right);

        FrameData* cur = window->getCur();

        if (blink_left && blink_right)
        {
            cur->meta()->eyeStateLeft(EyeState::closed);
            cur->meta()->eyeStateRight(EyeState::closed);
        }
        else
        {
            cur->meta()->eyeStateLeft(EyeState::open);
            cur->meta()->eyeStateRight(EyeState::open);
        }

        return true;
    }
};
