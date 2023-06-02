import Switch, { SwitchProps } from '@mui/material/Switch'
import { styled } from '@mui/material/styles'
import React from 'react'

const IOSSwitch = styled((props: SwitchProps) => (
  <Switch focusVisibleClassName='.Mui-focusVisible' disableRipple {...props} />
))(({ theme }) => ({
  width: 50,
  height: 26,
  padding: 0,
  '& .MuiSwitch-switchBase': {
    padding: 0,
    margin: 2,
    transitionDuration: '300ms',
    marginLeft: '3px',
    '&:hover': {
      boxShadow: '0px 0px 15px 6px #ccc'
    },

    '&.Mui-checked': {
      transform: 'translateX(22px)',
      color: '#fff',
      '&:hover': {
        boxShadow: '0px 0px 15px 6px #64c265'
      },
      '& + .MuiSwitch-track': {
        backgroundColor: theme.palette.mode === 'dark' ? '#2ECA45' : '#65C466',
        opacity: 1,
        border: 0
      },
      '&.Mui-disabled + .MuiSwitch-track': {
        opacity: 0.5
      }
    },
    '&.Mui-focusVisible .MuiSwitch-thumb': {
      color: '#33cf4d',
      border: '6px solid #fff'
    },
    '&.Mui-disabled .MuiSwitch-thumb': {
      color:
        theme.palette.mode === 'light'
          ? theme.palette.grey[100]
          : theme.palette.grey[600]
    },
    '&.Mui-disabled + .MuiSwitch-track': {
      opacity: theme.palette.mode === 'light' ? 0.7 : 0.3
    }
  },
  '& .MuiSwitch-thumb': {
    boxSizing: 'border-box',
    width: 22,
    height: 22
  },
  '& .MuiSwitch-track': {
    borderRadius: 26 / 2,
    backgroundColor: theme.palette.mode === 'light' ? '#E9E9EA' : '#39393D',
    opacity: 1,
    transition: theme.transitions.create(['background-color'], {
      duration: 500
    })
  }
}))

type Props = {
  handleChange: (event: React.ChangeEvent<HTMLInputElement>) => void
  checkSwitch: boolean
}

export default function CustomizedSwitch(props: Props) {
  const { handleChange, checkSwitch } = props

  return (
    <div className='container__switch-customize'>
      <span className='title'>Change information</span>
      <IOSSwitch
        className='switch'
        checked={checkSwitch}
        onChange={(event) => {
          handleChange(event)
        }}
      />
    </div>
  )
}
