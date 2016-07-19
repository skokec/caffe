#include <boost/thread.hpp>
#include <string>

#include "caffe/data_reader.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/parallel.hpp"
#include "caffe/util/blocking_queue.hpp"

#include <boost/chrono/system_clocks.hpp>

#include <chrono>
#include <mutex>
#include <condition_variable>

namespace caffe {

template<typename T>
class BlockingQueue<T>::sync {
 public:
  //mutable boost::mutex mutex_;
  //boost::condition_variable condition_;
	mutable std::mutex mutex_;
	std::condition_variable condition_;
};

template<typename T>
BlockingQueue<T>::BlockingQueue()
    : sync_(new sync()) {
}

template<typename T>
void BlockingQueue<T>::push(const T& t) {
  //boost::mutex::scoped_lock lock(sync_->mutex_);
  std::unique_lock<std::mutex> lock(sync_->mutex_);

  queue_.push(t);
  lock.unlock();
  sync_->condition_.notify_one();
}

template<typename T>
bool BlockingQueue<T>::try_pop(T* t) {
  //boost::mutex::scoped_lock lock(sync_->mutex_);
  std::unique_lock<std::mutex> lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  queue_.pop();
  return true;
}

template<typename T>
T BlockingQueue<T>::pop(const string& log_on_wait, const bool wait_indefinetly) {
  //boost::mutex::scoped_lock lock(sync_->mutex_);
  std::unique_lock<std::mutex> lock(sync_->mutex_);

  // do not perform infinity wait since if thread wants to stop interuption may not work
  // instead just wait a few sec and return 0 if nothing found - caller should make sure to check if
  // it needs to stop and call this method again if not; note, returing 0 is not correct if template is not pointer !!!
  while (queue_.empty()) {
  //if (queue_.empty()) {
    if (!log_on_wait.empty()) {
      LOG_EVERY_N(INFO, 1000)<< log_on_wait;
    }

    // use C++11 condition and chrono wait since chrono in boost is buggy
    if (sync_->condition_.wait_for(lock,std::chrono::seconds(1)) == std::cv_status::timeout)
    	if (!wait_indefinetly)
    		return 0;

    //sync_->condition_.wait(lock);
  }

  T t = queue_.front();
  queue_.pop();
  return t;
}

template<typename T>
bool BlockingQueue<T>::try_peek(T* t) {
  //boost::mutex::scoped_lock lock(sync_->mutex_);
  std::unique_lock<std::mutex> lock(sync_->mutex_);

  if (queue_.empty()) {
    return false;
  }

  *t = queue_.front();
  return true;
}

template<typename T>
T BlockingQueue<T>::peek(const bool wait_indefinetly) {
  //boost::mutex::scoped_lock lock(sync_->mutex_);
  std::unique_lock<std::mutex> lock(sync_->mutex_);

  // do not perform infinity wait since if thread wants to stop interuption may not work
  // instead just wait a few sec and return 0 if nothing found - caller should make sure to check if
  // it needs to stop and call this method again if not; note, returing 0 is not correct if template is not pointer !!!
  while (queue_.empty()) {
  //if (queue_.empty()) {
    //sync_->condition_.wait(lock);
     if (sync_->condition_.wait_for(lock,std::chrono::seconds(1)) == std::cv_status::timeout)
    	if (!wait_indefinetly)
    		return 0;
  }

  return queue_.front();
}

template<typename T>
size_t BlockingQueue<T>::size() const {
  //boost::mutex::scoped_lock lock(sync_->mutex_);
  std::unique_lock<std::mutex> lock(sync_->mutex_);
  return queue_.size();
}

template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;
template class BlockingQueue<Datum*>;
template class BlockingQueue<shared_ptr<DataReader::QueuePair> >;
template class BlockingQueue<P2PSync<float>*>;
template class BlockingQueue<P2PSync<double>*>;

}  // namespace caffe
