import SearchIcon from '@mui/icons-material/Search'
import React, { useState } from 'react'
import { useSelector } from 'react-redux'

import Button from '../../../components/ButtonCustomize'
import { RootState } from '../../../redux/configStore'
import { useAppDispatch } from '../../../redux/hooks'
import {
  clearListDepartmentId,
  getAllMedicalExaminationTimeAction,
  getListPythonDepartmentId
} from '../../../redux/slices/medicalExaminationSlice'

export default function Search() {
  const [inputSearch, setInputSearch] = useState('')
  const { arrMedicalExaminations } = useSelector(
    (state: RootState) => state.medicalExaminationReducer
  )

  const dispatch = useAppDispatch()

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = arrMedicalExaminations.filter((item) => {
      const { medicalExamination } = item
      return medicalExamination.title
        .toLowerCase()
        .includes(inputSearch.toLowerCase())
    })

    console.log(arrMedicalExaminations)
    const fetchApiGetListPythonDepartmentId = async () => {
      await dispatch(getListPythonDepartmentId(inputSearch.toLowerCase()))
    }
    if (inputSearch) {
      fetchApiGetListPythonDepartmentId()
    }
    dispatch(clearListDepartmentId())
    // dispatch(getAllMedicalExaminationTimeAction(data))
    // setInputSearch('')

    // console.log(data)
  }

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputSearch(event.target.value)
  }
  return (
    <>
      <form action='' className='form__search' onSubmit={handleSubmit}>
        <div className='container__input-search'>
          <SearchIcon className='icon__search' />
          <input
            className='input__search'
            placeholder='Search....'
            value={inputSearch}
            onChange={handleChange}
          />
          <Button text='Search' className='btn__search' type='submit' />
        </div>
      </form>
    </>
  )
}
