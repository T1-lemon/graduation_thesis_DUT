import DeleteIcon from '@mui/icons-material/Delete'
import IconButton from '@mui/material/IconButton'
import React, { useEffect, useRef, useState } from 'react'

import {
  addHoursToDate,
  getTimeZone,
  removeDuplicatesDate,
  sortDate
} from '../../../utils/date'

type Props = {
  indexTimeItem: number
  handleChange: (
    index: number,
    event: React.ChangeEvent<HTMLSelectElement>
  ) => void
  removeSelect: (index: number) => void
  isDisable: boolean
  arrStartTimeCorrect: Date[]
  arrEndTime: Date[]
  valueStartTime: string
  valueEndTime: string
  lengthSelectTime: number
}

export default function FormSelectTime(props: Props) {
  const {
    handleChange,
    removeSelect,
    indexTimeItem,
    isDisable,
    arrStartTimeCorrect,
    arrEndTime,
    valueStartTime,
    valueEndTime,
    lengthSelectTime
  } = props

  const [arrEndTimeCorrect, setArrEndTimeCorrect] = useState<Date[]>([])

  const [isSelectStartTime, setIsSelectStartTime] = useState<boolean>(false)

  // const getOptionsEndTime = (event: React.ChangeEvent<HTMLSelectElement>) => {
  //   const startTime: Date = new Date(event.target.value)
  //   const arrEndTimeTemporary: Date[] = [
  //     addHoursToDate(startTime, 30),
  //     addHoursToDate(startTime, 60)
  //   ]
  //   const arrEndTimeCorrect: Date[] = arrEndTimeTemporary.filter(
  //     (endTimeTemporary) => {
  //       const checkExist = arrEndTime.some(
  //         (endTime) => endTime.getTime() === endTimeTemporary.getTime()
  //       )
  //       return !checkExist
  //     }
  //   )

  //   setArrEndTimeCorrect(arrEndTimeCorrect)
  // }

  const getOptionsEndTime = (startTimeOption: string) => {
    const startTime: Date = new Date(startTimeOption)
    const arrEndTimeTemporary: Date[] = [
      addHoursToDate(startTime, 30),
      addHoursToDate(startTime, 60)
    ]
    // console.log('arrEndTimeTemporary', arrEndTimeTemporary)
    const arrEndTimeCorrect: Date[] = arrEndTimeTemporary.filter(
      (endTimeTemporary) => {
        const checkExist = arrEndTime.some(
          (endTime) => endTime.getTime() === endTimeTemporary.getTime()
        )
        return !checkExist
      }
    )

    setArrEndTimeCorrect(arrEndTimeCorrect)
  }

  useEffect(() => {
    setIsSelectStartTime(false)
    if (valueStartTime) getOptionsEndTime(valueStartTime)
  }, [arrEndTime])

  const [startTime, setStartTime] = useState<string>(valueStartTime)
  const [endTime, setEndTime] = useState<string>(valueEndTime)

  // console.log(
  //   `ref arrStartTimeCorrect select ${indexTimeItem}`,
  //   arrStartTimeCorrectRef.current
  // )
  // console.log(` arrEndTimeCorrect select ${indexTimeItem}`, arrEndTimeCorrect)

  // console.log(` start time select ${indexTimeItem}`, startTime)

  const renderValueStartTimeOption = () => {
    const checkExist = arrStartTimeCorrect.some(
      (startTimeItem) =>
        startTimeItem.getTime() === new Date(valueStartTime).getTime()
    )
    // console.log(`value start time select ${indexTimeItem}`, valueStartTime)

    return !checkExist && valueStartTime ? (
      <option value={valueStartTime}>
        {valueStartTime ? getTimeZone(valueStartTime) : ''}
      </option>
    ) : (
      ''
    )
  }

  const renderValueEndTimeOption = () => {
    const checkExist = arrEndTimeCorrect.some(
      (endTimeItem) =>
        endTimeItem.getTime() === new Date(valueEndTime).getTime()
    )

    return !checkExist && valueEndTime && !isSelectStartTime ? (
      <option value={valueEndTime}>
        {valueEndTime ? getTimeZone(valueEndTime) : ''}
      </option>
    ) : (
      ''
    )
  }

  const renderOptionsStartTimeCorrect = () => {
    const arrOptions: Date[] = removeDuplicatesDate(arrStartTimeCorrect)
    const arrOptionsSort: Date[] = sortDate(arrOptions)
    return arrOptionsSort.map((item, index) => {
      return (
        <option key={index} value={`${item}`}>
          {getTimeZone(item)}
        </option>
      )
    })
  }

  return (
    <div className='form__row'>
      <div className='form__group'>
        <label>Start Time</label>
        <select
          className='form__select'
          onChange={(event) => {
            handleChange(indexTimeItem, event)
            getOptionsEndTime(event.target.value)
            setIsSelectStartTime(true)
            // setStartTime(event.target.value)
          }}
          name='startTime'
          disabled={isDisable}
          value={valueStartTime}
        >
          <option value=''>-</option>
          {renderOptionsStartTimeCorrect()}

          {renderValueStartTimeOption()}
          {/* {valueStartTime ? (
            <option value={valueStartTime}>
              {getTimeZone(valueStartTime)}
            </option>
          ) : (
            ''
          )} */}
        </select>
      </div>

      <div className='form__group'>
        <label>End Time</label>
        <select
          className='form__select'
          onChange={(event) => {
            handleChange(indexTimeItem, event)
            // setEndTime(event.target.value)
          }}
          name='endTime'
          disabled={isDisable}
          value={valueEndTime}
        >
          <option value=''>-</option>
          {arrEndTimeCorrect.map((item, index) => {
            return (
              <option key={index} value={`${item}`}>
                {getTimeZone(item)}
              </option>
            )
          })}

          {/* {valueEndTime ? (
            <option value={valueEndTime}>{getTimeZone(valueEndTime)}</option>
          ) : (
            ''
          )} */}
          {renderValueEndTimeOption()}
        </select>
      </div>

      {lengthSelectTime === 1 && !isDisable ? (
        ''
      ) : (
        <IconButton
          aria-label='delete'
          onClick={() => {
            removeSelect(indexTimeItem)
            // setStartTime('')
            // setEndTime('')
          }}
        >
          <DeleteIcon />
        </IconButton>
      )}
    </div>
  )
}
