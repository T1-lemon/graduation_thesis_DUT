import CloseIcon from '@mui/icons-material/Close'
import Box from '@mui/material/Box'
import IconButton from '@mui/material/IconButton'
import Modal from '@mui/material/Modal'
import React, { useEffect, useState } from 'react'

import ButtonCustomize from '../../../components/ButtonCustomize'
import { MODAL_ACTION_CLOSE, SUBMIT_MODAL_EDIT } from '../../../utils/contant'
import {
  addHoursToDate,
  checkExistArrayDate,
  getTimeZone,
  sortDate
} from '../../../utils/date'

type Props = {
  openModalEdit: boolean
  onAction: (type: string) => void
  handleChange: (event: React.ChangeEvent<HTMLSelectElement>) => void
  valueStartTime: string
  valueEndTime: string
  arrStartTimeCorrect: Date[]
  arrEndTime: Date[]
  arrEndTimeCorrectOriginal: Date[]
}

export default function ModalEditTime(props: Props) {
  const {
    openModalEdit,
    onAction,
    handleChange,
    valueStartTime,
    valueEndTime,
    arrStartTimeCorrect,
    arrEndTime,
    arrEndTimeCorrectOriginal
  } = props

  const [arrEndTimeCorrect, setArrEndTimeCorrect] = useState<Date[]>(
    arrEndTimeCorrectOriginal
  )

  const getOptionsEndTime = (startTimeOption: string) => {
    const startTime: Date = new Date(startTimeOption)

    const arrEndTimeTemporary: Date[] = [
      addHoursToDate(startTime, 30),
      addHoursToDate(startTime, 60)
    ]

    const arrEndTimeCorrect: Date[] = arrEndTimeTemporary.filter(
      (endTimeTemporary) => {
        return !checkExistArrayDate(arrEndTime, endTimeTemporary)
      }
    )

    setArrEndTimeCorrect(arrEndTimeCorrect)
  }

  const renderOptionsStartTimeCorrect = () => {
    return sortDate(arrStartTimeCorrect).map((item, index) => {
      return (
        <option key={index} value={`${item}`}>
          {getTimeZone(item)}
        </option>
      )
    })
  }

  useEffect(() => {
    setArrEndTimeCorrect(arrEndTimeCorrectOriginal)
  }, [arrEndTimeCorrectOriginal])

  return (
    <>
      <Modal
        open={openModalEdit}
        onClose={() => {
          onAction(MODAL_ACTION_CLOSE)
        }}
        aria-labelledby='modal-modal-title'
        aria-describedby='modal-modal-description'
      >
        <Box className='container__modal-schedule'>
          <div className='modal__header'>
            <div className='title'>
              <h2>Edit Time Slots</h2>
            </div>
            <IconButton
              onClick={() => {
                onAction(MODAL_ACTION_CLOSE)
              }}
              className='icon__close'
            >
              <CloseIcon />
            </IconButton>
          </div>
          <div className='modal__body'>
            <form
              action=''
              onSubmit={(event: React.FormEvent<HTMLFormElement>) => {
                event.preventDefault()
                onAction(SUBMIT_MODAL_EDIT)
              }}
            >
              <div className='container__select__time'>
                <div className='form__row'>
                  <div className='form__group'>
                    <label>Start Time</label>
                    <select
                      className='form__select'
                      name='startTime'
                      onChange={(event) => {
                        handleChange(event)
                        getOptionsEndTime(event.target.value)
                      }}
                      value={`${new Date(valueStartTime)}`}
                    >
                      <option value=''>-</option>
                      {renderOptionsStartTimeCorrect()}
                    </select>
                  </div>

                  <div className='form__group'>
                    <label>End Time</label>
                    <select
                      className='form__select'
                      name='endTime'
                      onChange={(event) => {
                        handleChange(event)
                      }}
                      value={`${new Date(valueEndTime)}`}
                    >
                      <option value=''>-</option>
                      {arrEndTimeCorrect.map((item, index) => {
                        return (
                          <option key={index} value={`${item}`}>
                            {getTimeZone(item)}
                          </option>
                        )
                      })}
                    </select>
                  </div>
                </div>
              </div>
              <ButtonCustomize
                className='btn__save'
                type='submit'
                text='Save Change'
              />
            </form>
          </div>
        </Box>
      </Modal>
    </>
  )
}
