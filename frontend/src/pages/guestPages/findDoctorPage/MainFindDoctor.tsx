import FmdBadIcon from '@mui/icons-material/FmdBad'
import Grid from '@mui/material/Grid'
import React, { useEffect } from 'react'
import { useSelector } from 'react-redux'

import '../../../assets/css/pages/guestPage/findDoctorPage/find_doctor.css'
import { IMedicalExaminationTime } from '../../../interface/MedicalExaminationInterfaces'
import { RootState } from '../../../redux/configStore'
import { useAppDispatch } from '../../../redux/hooks'
import { clearListDepartmentId, getAllMedicalExaminationTimeThunk } from '../../../redux/slices/medicalExaminationSlice'
import FilterDoctor from './FilterDoctor'
import InformationAppointment from './InformationAppointment'
import Search from './Search'

export default function MainFindDoctor() {
  const dispatch = useAppDispatch()
  const { arrMedicalExaminations, arrDeparmentId } = useSelector(
    (state: RootState) => state.medicalExaminationReducer
  )
  const getMedicalExaminationTimeApi = async () => {
    await dispatch(getAllMedicalExaminationTimeThunk())
  }

  useEffect(() => {
    getMedicalExaminationTimeApi()
    dispatch(clearListDepartmentId())
  }, [])

  const renderMedicalExamination = () => {
    if (arrMedicalExaminations.length === 0)
      return (
        <div className='box__filter__not__found'>
          <div className='filter__not__found-view'>
            <FmdBadIcon />
            <p className='filter__not__found-text'>
              No matches were found, please choose again
            </p>
          </div>
        </div>
      )
    if (arrDeparmentId.length === 0) {
      return arrMedicalExaminations.map(
        (medicalExamination: IMedicalExaminationTime, index: number) => {
          return (
            <InformationAppointment
              key={index}
              examinationAndTime={medicalExamination}
            />
          )
        }
      )
    }
    let count = 0
    return (
      <>
        {arrMedicalExaminations.map(
          (medicalExamination: IMedicalExaminationTime, index: number) => {
            const departmentId =
              medicalExamination.medicalExamination.department.id
            if (arrDeparmentId.includes(departmentId)) {
              count++
              return (
                <InformationAppointment
                  key={index}
                  examinationAndTime={medicalExamination}
                />
              )
            }
          }
        )}
        <div>
          {count === 0 ? (
            <>
              <div className='box__filter__not__found'>
                <div className='filter__not__found-view'>
                  <FmdBadIcon />
                  <p className='filter__not__found-text'>
                    No matches were found, please choose again
                  </p>
                </div>
              </div>
            </>
          ) : (
            <></>
          )}
        </div>
      </>
    )
  }
  return (
    <section className='container__home-doctor'>
      <div className='search__doctor-image'>
        <div className='container__search-doctor'>
          <img
            src='http://azim.commonsupport.com/Docpro/assets/images/banner/banner-image-1.png'
            alt=''
          />
          <h1 className='title'>Search Doctor, Make an Appointment</h1>
          <Search />
        </div>
      </div>
      <section className='infor__appointment__container'>
        <Grid container spacing={5} className='grid__container'>
          <Grid item xs={3} className='grid__filter'>
            <FilterDoctor />
          </Grid>
          <Grid item xs={9} className='information__container'>
            {renderMedicalExamination()}
          </Grid>
        </Grid>
      </section>
    </section>
  )
}
