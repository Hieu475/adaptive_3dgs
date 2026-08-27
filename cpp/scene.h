#pragma once
#include <vector>
#include <string>

namespace adaptive3dgs {

/**
 * @brief Manages the 3D scene data (Gaussians, cameras).
 */
class Scene {
public:
    Scene();
    ~Scene();

    void load(const std::string& path);
    void updateGaussians();
    
    // TODO: Add accessors and modifiers
};

} // namespace adaptive3dgs
