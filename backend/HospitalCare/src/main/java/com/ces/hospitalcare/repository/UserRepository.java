package com.ces.hospitalcare.repository;
import com.ces.hospitalcare.entity.UserEntity;
import com.ces.hospitalcare.util.Role;
import java.util.List;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserRepository extends JpaRepository<UserEntity, Long> {
  Optional<UserEntity> findByEmail(String email);

  List<UserEntity> getAllByRole(Role role);

  Optional<UserEntity> findByIdAndRole(Long id, Role role);
}

