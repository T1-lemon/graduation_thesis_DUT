import Visibility from '@mui/icons-material/Visibility'
import VisibilityOff from '@mui/icons-material/VisibilityOff'
import {
  FormControlLabel,
  Grid,
  IconButton,
  InputAdornment,
  InputLabel,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
  TextField
} from '@mui/material'
import { FormikErrors } from 'formik'
import React, { useState } from 'react'

import { FormUserProfileValues } from '../../interface/UsersInterface'
import {
  FormLoginValues,
  FormRegisterValues
} from '../../interface/ValidateInterface'
import { useAppSelector } from '../../redux/hooks'
import { CHECK_NAME_MATCH_REGEX } from '../validateInform'

type propsFieldInput = {
  fieldId: string
  fieldLable: string
  fieldAutoFocus?: boolean
  onFieldChange: any
  values: FormLoginValues | FormRegisterValues | FormUserProfileValues
  errors: FormikErrors<
    FormLoginValues | FormRegisterValues | FormUserProfileValues
  >
}

const VIETNAMFAG = require('../../assets/img/vietnam_flag.png')

function FieldInputFormik(props: propsFieldInput) {
  const { fieldId, fieldLable, fieldAutoFocus, onFieldChange, values, errors } =
    props
  const [showPassword, setShowPassword] = useState(false)
  const handleClickShowPassword = () => setShowPassword((show) => !show)
  const handleMouseDownPassword = (
    event: React.MouseEvent<HTMLButtonElement>
  ) => {
    event.preventDefault()
  }
  const { listDepartment } = useAppSelector((state) => state.admin)

  const fieldKey = fieldId as keyof typeof values
  return (
    <>
      {fieldId === 'gender' ? (
        <Grid item xs={12} className='auth__div-radio'>
          <InputLabel shrink htmlFor={fieldId} className='lable__input'>
            {fieldLable}
            <span>*</span>
          </InputLabel>
          <RadioGroup
            row
            aria-labelledby='demo-radio-buttons-group-label'
            defaultValue='female'
            name='radio-buttons-group'
            value={values[fieldKey]}
            onChange={onFieldChange}
          >
            <FormControlLabel
              value='female'
              name='gender'
              control={<Radio />}
              label='Female'
            />
            <FormControlLabel
              value='male'
              name='gender'
              control={<Radio />}
              label='Male'
            />
          </RadioGroup>
        </Grid>
      ) : (
        <></>
      )}

      {fieldId === 'departmentId' ? (
        <Grid item xs={12}>
          <InputLabel shrink htmlFor={fieldId} className='lable__input'>
            {fieldLable}
            <span>*</span>
          </InputLabel>
          <Select
            labelId='demo-select-small-label'
            name={fieldId}
            value={values[fieldKey] ? values[fieldKey] : ''}
            onChange={onFieldChange}
          >
            {listDepartment ? (
              listDepartment.map((department) => {
                return (
                  <MenuItem value={`${department.id}`} key={`${department.id}`}>
                    {`${department.name}`}
                  </MenuItem>
                )
              })
            ) : (
              <MenuItem value=''>
                <em>None</em>
              </MenuItem>
            )}
          </Select>
        </Grid>
      ) : (
        <></>
      )}

      {fieldId !== 'departmentId' && fieldId !== 'gender' ? (
        <Grid
          item
          xs={12}
          sm={fieldId === 'firstName' || fieldId === 'lastName' ? 6 : 0}
        >
          <InputLabel shrink htmlFor={fieldId} className='lable__input'>
            {fieldLable}
            <span>*</span>
          </InputLabel>
          <TextField
            error={errors[fieldKey] ? true : false}
            fullWidth
            id={fieldId}
            autoFocus={fieldAutoFocus}
            size='small'
            className={`auth__input ${
              errors[fieldKey] === CHECK_NAME_MATCH_REGEX
                ? 'auth__input--extra'
                : ''
            }`}
            value={values[fieldKey]}
            onChange={onFieldChange}
            helperText={errors[fieldKey]}
            type={
              fieldId === 'password' || fieldId === 'confirmPassword'
                ? showPassword
                  ? 'text'
                  : 'password'
                : 'text'
            }
            InputProps={
              fieldId === 'phoneNumber'
                ? {
                    startAdornment: (
                      <InputAdornment position='start'>
                        <img
                          src={VIETNAMFAG}
                          alt='#'
                          className='auth__vn--flag'
                        />
                      </InputAdornment>
                    )
                  }
                : fieldId === 'password' || fieldId === 'confirmPassword'
                ? {
                    endAdornment: (
                      <InputAdornment position='end'>
                        <IconButton
                          aria-label='toggle password visibility'
                          onClick={handleClickShowPassword}
                          onMouseDown={handleMouseDownPassword}
                          edge='end'
                        >
                          {showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    )
                  }
                : {}
            }
          />
        </Grid>
      ) : (
        <></>
      )}
    </>
  )
}

export default FieldInputFormik
