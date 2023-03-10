#pragma once

#include "mesh.h"
#include "time_system.h"

class RectCloth : public Mesh {
 public:

  /// constructor

  RectCloth(Float cloth_weight,
            const UVec2& mass_dim,
            Float dx_local,
            Float stiffness, Float damping_ratio);

  RectCloth(const RectCloth&) = default;
  RectCloth(RectCloth&&) = default;
  RectCloth& operator=(const RectCloth&) = default;
  RectCloth& operator=(RectCloth&&) = default;
  virtual ~RectCloth() override = default;



  /// interfaces

  bool SetMassFixedOrNot(int iw, int ih, bool fixed_or_not);

  virtual void FixedUpdate() override;

    static Mat4 CamearTransformMat;
    static int isFirstFrame;

private:
  static constexpr unsigned simulation_steps_per_fixed_update_time = 20;
  static constexpr Float fixed_delta_time = Time::fixed_delta_time / Float(simulation_steps_per_fixed_update_time);

  /// 30x40
  UVec2 mass_dim;
  /// 每一个质点的质量
  Float mass_weight;

  Float dx_local;

  /// 胡克定律中的k
  Float stiffness;
  /// F固定系数
  Float damping_ratio;

  std::vector<bool> is_fixed_masses;
  std::vector<Vec3> local_or_world_positions;
  std::vector<Vec3> world_velocities;
  std::vector<Vec3> world_accelerations;

  Vec3 sphereCenter=Vec3(1, -1.8, 0.3);
//  Vec3 sphereCenter=Vec3(3.5, -1.8, 0.3);

bool isPickedPoint=false;


  /// force computation

  [[nodiscard]] Vec3 ComputeHookeForce(int iw_this, int ih_this,
                                       int iw_that, int ih_that,
                                       Float dx_world) const;

  [[nodiscard]] Vec3 ComputeSpringForce(int iw, int ih) const;



  /// simulation pipeline

  void LocalToWorldPositions();

  void ComputeAccelerations();

  void ComputeVelocities();

  void ComputePositions();

  void WorldToLocalPositions();

  void Simulate(unsigned num_steps);



  /// rendering

  void UpdateMeshVertices();



  /// supporting methods

  [[nodiscard]] size_t Get1DIndex(int iw, int ih) const;
  bool Get1DIndex(int iw, int ih, size_t& idx) const;

    Vec3 ComputeGravityForce() const;
};
