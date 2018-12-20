/* Copyright (C) 2016-2018 Alibaba Group Holding Limited

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef PS_COMMON_RD_LOCK_H_
#define PS_COMMON_RD_LOCK_H_

#include <pthread.h>

namespace ps {

class ReadWriteLock
{
private:
    ReadWriteLock(const ReadWriteLock&);
    ReadWriteLock& operator = (const ReadWriteLock&);
public:
    enum Mode {
        PREFER_READER,
        PREFER_WRITER
    };

    ReadWriteLock(Mode mode = PREFER_READER) {
        pthread_rwlockattr_t attr;
        pthread_rwlockattr_init(&attr);
#ifdef __linux__
        switch (mode)
        {
        case PREFER_WRITER:
            pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
            break;
        case PREFER_READER:
            pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_READER_NP);
            break;
        default:
            break;
        }
#endif
        pthread_rwlock_init(&_lock, &attr);
    }

    ~ReadWriteLock() {
        pthread_rwlock_destroy(&_lock);
    }

    int rdlock() {
        return pthread_rwlock_rdlock(&_lock);
    }

    int wrlock() {
        return pthread_rwlock_wrlock(&_lock);
    }

    int tryrdlock() {
        return pthread_rwlock_tryrdlock(&_lock);
    }

    int trywrlock() {
        return pthread_rwlock_trywrlock(&_lock);
    }

    int unlock() {
        return pthread_rwlock_unlock(&_lock);
    }

protected:
    pthread_rwlock_t _lock;
};

class ScopedReadLock
{
private:
    ReadWriteLock &_lock;
private:
    ScopedReadLock(const ScopedReadLock&);
    ScopedReadLock& operator = (const ScopedReadLock&);
public:
    explicit ScopedReadLock(ReadWriteLock &lock) 
        : _lock(lock) 
    {
        _lock.rdlock();
    }
    ~ScopedReadLock() {
        _lock.unlock();
    }
};

class ScopedWriteLock
{
private:
    ReadWriteLock &_lock;
private:
    ScopedWriteLock(const ScopedWriteLock&);
    ScopedWriteLock& operator = (const ScopedWriteLock&);
public:
    explicit ScopedWriteLock(ReadWriteLock &lock) 
        : _lock(lock) 
    {
        _lock.wrlock();
    }
    ~ScopedWriteLock() {
        _lock.unlock();
    }
};

class ScopedReadWriteLock
{
private:
    ReadWriteLock& _lock;
    char _mode;
private:
    ScopedReadWriteLock(const ScopedReadWriteLock&);
    ScopedReadWriteLock& operator = (const ScopedReadWriteLock&);

public:
    explicit ScopedReadWriteLock(ReadWriteLock& lock, const char mode) 
        : _lock(lock), _mode(mode)
    {
        if (_mode == 'r' || _mode == 'R') {
            _lock.rdlock();
        } else if (_mode == 'w' || _mode == 'W') {
            _lock.wrlock();
        }
    }
    
    ~ScopedReadWriteLock()
    {
        if (_mode == 'r' || _mode == 'R' 
            || _mode == 'w' || _mode == 'W') 
        { 
            _lock.unlock();
        }
    }
};

}

#endif

