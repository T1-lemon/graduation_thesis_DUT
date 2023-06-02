import { useFormik } from 'formik'
import { useEffect, useState } from 'react'
import CurrencyInput from 'react-currency-input-field'

import Button from '../../../components/ButtonCustomize'
import { ProgressListener } from '../../../components/Progress'
import { ICategory } from '../../../interface/CategoryInterfaces'
import { IMedicalExaminationFilter } from '../../../interface/MedicalExaminationInterfaces'
import { useAppDispatch } from '../../../redux/hooks'
import {
  filterMedicalExaminationTimeThunkByCategory,
  filterMedicalExaminationTimeThunkByCategoryAndPrice
} from '../../../redux/slices/medicalExaminationSlice'
import { getAllCategoryService } from '../../../services/categoryService'

type checkOptions = {
  [key: string]: boolean
}

export default function FilterDoctor() {
  const dispatch = useAppDispatch()

  const [categories, setCategories] = useState<ICategory[]>()
  const getAllCategories = async () => {
    const arrCategory = await getAllCategoryService()
    setCategories(arrCategory)
  }

  useEffect(() => {
    getAllCategories()
  }, [])

  const createInitialCategoryCheck = () => {
    const arrNameCategory = categories?.map((item) => item.name)
    const initialCategoryCheck: checkOptions = {}
    arrNameCategory?.forEach((item) => {
      initialCategoryCheck[item] = false
    })
    return initialCategoryCheck
  }

  const [categoryCheck, setCategoryCheck] = useState<checkOptions>(
    createInitialCategoryCheck()
  )

  const formik = useFormik<IMedicalExaminationFilter>({
    initialValues: {
      category: [],
      minPrice: 0,
      maxPrice: 999999999999
    },
    onSubmit: (values) => {
      let { category, minPrice, maxPrice } = values
      ProgressListener.emit('start')
      setTimeout(() => {
        dispatch(
          filterMedicalExaminationTimeThunkByCategoryAndPrice(
            category,
            minPrice,
            maxPrice
          )
        )
        ProgressListener.emit('stop')
      }, 2000)
    }
  })

  const { handleChange, handleSubmit, setFieldValue, values } = formik
  const handleChangeCheckbox = (event: React.ChangeEvent<HTMLInputElement>) => {
    handleChange(event)
    const objCategoryCheck = { ...categoryCheck }
    objCategoryCheck[event.target.value] = event.target.checked
    const arrCategory = Object.keys(objCategoryCheck).filter(
      (item) => objCategoryCheck[item]
    )
    ProgressListener.emit('start')
    setTimeout(() => {
      dispatch(
        filterMedicalExaminationTimeThunkByCategoryAndPrice(
          arrCategory,
          values.minPrice,
          values.maxPrice
        )
      )
      ProgressListener.emit('stop')
    }, 2000)

    setCategoryCheck(objCategoryCheck)
  }

  return (
    <div className='filter__container'>
      <form action='' className='form__filter' onSubmit={handleSubmit}>
        <div className='select__form-object filter__doctor'>
          <div className='filter__header'>
            <h3>Categories</h3>
          </div>
          <div className='filter__widget filter__checkbox'>
            {categories?.map((category, index) => {
              return (
                <div key={index}>
                  <label className='custom_check'>
                    <input
                      onChange={handleChangeCheckbox}
                      type='checkbox'
                      name='category'
                      value={category.name}
                    />
                    <span className='checkmark' /> {category.name}
                  </label>
                </div>
              )
            })}
          </div>
        </div>

        <div className='form__price__range filter__doctor'>
          <div className='filter__header'>
            <h3> Examination Price Range</h3>
          </div>
          <div className='container__price__range filter__widget'>
            <CurrencyInput
              placeholder='$0'
              name='minPrice'
              className='input__price min__price'
              intlConfig={{ locale: 'en-US', currency: 'USD' }}
              onValueChange={(value) => {
                !value
                  ? setFieldValue('minPrice', 0)
                  : setFieldValue('minPrice', Number(value))
              }}
              value={values.minPrice === 0 ? '' : values.minPrice}
            />
            <span className='separator__symbol'>-</span>

            <CurrencyInput
              placeholder='$999,999,999,999'
              onValueChange={(value) => {
                !value
                  ? setFieldValue('maxPrice', 999999999999)
                  : setFieldValue('maxPrice', Number(value))
              }}
              name='maxPrice'
              className='input__price max__price'
              intlConfig={{ locale: 'en-US', currency: 'USD' }}
              value={values.maxPrice === 999999999999 ? '' : values.maxPrice}
            />
          </div>
          <Button type='submit' text='Apply' className='btn__apply' />
        </div>
      </form>
    </div>
  )
}
