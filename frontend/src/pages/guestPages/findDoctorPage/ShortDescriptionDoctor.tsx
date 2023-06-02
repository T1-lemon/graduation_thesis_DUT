import LocalHospitalIcon from '@mui/icons-material/LocalHospital'
import React from 'react'
import Avatar from 'react-avatar'
import { NavLink } from 'react-router-dom'

import { IMedicalExamination } from '../../../interface/MedicalExaminationInterfaces'

type Props = {
  medical: IMedicalExamination
}

export default function ShortDescriptionDoctor(props: Props) {
  const { medical } = props

  return (
    <>
      <div className='container__description'>
        <div className=''>
          {medical.image ? (
            <img className='img__doctor' src={medical.image} alt='' />
          ) : (
            <Avatar facebookId='100008343750912' size='120' round={true} />
          )}

          <div className='btn__detail'>
            <NavLink
              className='link__see-more'
              to={`/examination/detail/${medical?.id}`}
            >
              See detail
            </NavLink>
          </div>
        </div>
        <div className='container__infor'>
          <NavLink
            to={`/examination/detail/${medical?.id}`}
            className='title__doctor'
          >
            {medical.title}
          </NavLink>
          <div className='container__short-description'>
            <p>{medical?.shortDescription.split('\n')[0]}</p>
            <div className='department'>
              <LocalHospitalIcon className='icon__department'/>
              <span>{medical?.department.name}</span>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
