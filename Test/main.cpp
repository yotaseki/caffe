#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
using namespace std;
namespace fs = boost::filesystem;

int CheckAnswer(int &true_count, int &TP, int &TN, int &FP, int &FN, string teacher_dir, string predict_dir, string filename)
{
	cv::Mat p = imread(predict_dir + filename,cv::IMREAD_GRAYSCALE);
	cv::Mat t = imread(teacher_dir + filename,cv::IMREAD_GRAYSCALE);
	if(p.empty() || t.empty())
	{
		return 1;
	}
	for(int x=0; x<p.cols; x++)
	{
		for(int y=0; y<p.rows; y++)
		{
			bool P = p.at<unsigned char>(y,x) > 0;
			bool T = t.at<unsigned char>(y,x) > 0;
			if(P)
				true_count++;
			if(P && T)
				TP++;
			else if(!T && P)
				FP++;
			else if(T && !P)
				FN++;
			else
				TN++;
		}
	}
	return 0;
}

int main(int argc, char **argv)
{
	ofstream ofs("output.csv");
	int filecount = 0;
	int true_count = 0;
	int TP=0;
	int TN=0;
	int FP=0;
	int FN=0;
	const fs::path teacher_dir(argv[1]);
	const fs::path predict_dir(argv[2]);
	BOOST_FOREACH(const fs::path& p, std::make_pair(fs::directory_iterator(teacher_dir),fs::directory_iterator())) 
	{
		if (!fs::is_directory(p))
		{
			fs::path extension = p.extension();
			if(extension.generic_string() == ".png")
			{
				if(!CheckAnswer(true_count,TP,TN,FP,FN, teacher_dir.c_str(), predict_dir.c_str(), p.filename().c_str()))
				{
					cout << p.filename() << endl;
					filecount++;
				}
			}
		}
	}
	//int pixels = filecount * 320 * 240;
	ofs << filecount << "," << true_count<< ","<< TP << "," << TN << "," << FP << "," << FN << endl;
	ofs.close();
	return 0;
}

