#include "runtime.h"

namespace adaptive3dgs {

Runtime::Runtime() : is_running_(false) {}

Runtime::~Runtime() {}

void Runtime::initialize() {
    // TODO: Initialize systems
}

void Runtime::step() {
    // TODO: Process one frame
}

void Runtime::run() {
    is_running_ = true;
    while (is_running_) {
        step();
    }
}

} // namespace adaptive3dgs
