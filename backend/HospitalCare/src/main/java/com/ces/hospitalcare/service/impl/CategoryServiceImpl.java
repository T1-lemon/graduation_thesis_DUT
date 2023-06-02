package com.ces.hospitalcare.service.impl;
import com.ces.hospitalcare.dto.CategoryDTO;
import com.ces.hospitalcare.entity.CategoryEntity;
import com.ces.hospitalcare.repository.CategoryRepository;
import com.ces.hospitalcare.service.ICategoryService;
import java.util.ArrayList;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class CategoryServiceImpl implements ICategoryService {
  @Autowired
  private CategoryRepository categoryRepository;

  @Autowired
  private ModelMapper mapper;

  public List<CategoryDTO> covertToListDTO(List<CategoryEntity> listCategoryEntity) {
    List<CategoryDTO> listCategoryDTO = new ArrayList<>();

    for (CategoryEntity entity : listCategoryEntity) {

      CategoryDTO dto = mapper.map(entity, CategoryDTO.class);
      listCategoryDTO.add(dto);
    }

    return listCategoryDTO;
  }

  @Override
  public List<CategoryDTO> getAllCategory() {
    List<CategoryEntity> listCategoryEntity = categoryRepository.findAll();

    return covertToListDTO(listCategoryEntity);
  }
}
